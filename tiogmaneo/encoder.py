# ----------------------------------------------------------------------------
#  TiOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of TiOgmaNeo is licensed to you under the terms described
#  in the TIOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import taichi as ti
import taichi.math as tm
import math
import io
import struct
from dataclasses import dataclass
from .helpers import *

@dataclass
class EncoderVisibleLayerDesc:
    size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
    radius: int = 2
    importance: float = 1.0

@ti.data_oriented
class EncoderVisibleLayer:
    weights: ti.Field
    usages: ti.Field
    reconstruction: ti.Field

    size: (int, int, int, int)
    radius: int
    importance: float
    h_to_v: tm.vec2
    v_to_h: tm.vec2
    reverse_radii: tm.ivec2

    def __init__(self, hidden_size: (int, int, int), desc: EncoderVisibleLayerDesc):
        self.size = desc.size
        self.radius = desc.radius
        self.importance = desc.importance

        diam = self.radius * 2 + 1

        self.weights = ti.field(param_type, shape=(hidden_size[0], hidden_size[1], hidden_size[2], diam, diam, self.size[2], self.size[3]))

        self.usages = ti.field(usage_type, shape=self.weights.shape)
        self.usages.fill(0)

        self.reconstruction = ti.field(param_type, shape=self.size)

        self.h_to_v = tm.vec2(self.size[0] / hidden_size[0], self.size[1] / hidden_size[1])
        self.v_to_h = tm.vec2(hidden_size[0] / self.size[0], hidden_size[1] / self.size[1])
        self.reverse_radii = tm.ivec2(math.ceil(self.v_to_h.x * diam * 0.5), math.ceil(self.v_to_h.y * diam * 0.5))

    # Initialization
    @ti.kernel
    def init_random(self):
        for hx, hy, hz, ox, oy, vz, vt in self.weights:
            self.weights[hx, hy, hz, ox, oy, vz, vt] = ti.cast(ti.random(), param_type)

    # Stepping
    @ti.kernel
    def accum_activations(self, hidden_size: tm.ivec3, vt_start: int, visible_states: ti.template(), activations: ti.template()):
        for hx, hy, hz in ti.ndrange(hidden_size.x, hidden_size.y, hidden_size.z):
            h_pos = tm.ivec2(hx, hy)

            v_center = tm.ivec2((h_pos.x + 0.5) * self.h_to_v.x, (h_pos.y + 0.5) * self.h_to_v.y)
            
            offset_start = v_center - self.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.size[0], v_center.x + 1 + self.radius), tm.min(self.size[1], v_center.y + 1 + self.radius))

            it_size = it_end - it_start

            s = 0.0
            count = it_size.x * it_size.y

            for ox, oy in ti.ndrange(it_size.x, it_size.y):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vt in range(self.size[3]):
                    visible_state = visible_states[v_pos.x, v_pos.y, (vt_start + vt) % self.size[3]]

                    s += self.weights[hx, hy, hz, ox, oy, visible_state, vt]

            activations[hx, hy, hz] += ti.cast(s / count, param_type)

    @ti.kernel
    def accum_gates(self, hidden_size: tm.ivec3, hidden_states: ti.template(), hidden_gates: ti.template()):
        for hx, hy in ti.ndrange(hidden_size.x, hidden_size.y):
            h_pos = tm.ivec2(hx, hy)

            v_center = tm.ivec2((h_pos.x + 0.5) * self.h_to_v.x, (h_pos.y + 0.5) * self.h_to_v.y)
            
            offset_start = v_center - self.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.size[0], v_center.x + 1 + self.radius), tm.min(self.size[1], v_center.y + 1 + self.radius))

            it_size = it_end - it_start

            hidden_state = hidden_states[hx, hy]

            s = 0
            count = it_size.x * it_size.y * self.size[2] * self.size[3]

            for ox, oy in ti.ndrange(it_size.x, it_size.y):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vz in range(self.size[2]):
                    for vt in range(self.size[3]):
                        s += self.usages[hx, hy, hidden_state, ox, oy, vz, vt]

            hidden_gates[hx, hy] += ti.cast(s / count, param_type)

    @ti.kernel
    def learn(self, hidden_size: tm.ivec3, vt_start: int, hidden_states: ti.template(), visible_states: ti.template(), hidden_gates: ti.template(), lr: float):
        for vx, vy, vt in ti.ndrange(self.size[0], self.size[1], self.size[3]):
            visible_state = visible_states[vx, vy, (vt_start + vt) % self.size[3]]

            v_pos = tm.ivec2(vx, vy)

            h_center = tm.ivec2((v_pos.x + 0.5) * self.v_to_h.x, (v_pos.y + 0.5) * self.v_to_h.y)
            
            offset_start = h_center - self.reverse_radii

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(hidden_size.x, h_center.x + 1 + self.reverse_radii.x), tm.min(hidden_size.y, h_center.y + 1 + self.reverse_radii.y))

            it_size = it_end - it_start

            count = it_size.x * it_size.y

            max_index = 0
            max_activation = limit_min

            # Reconstruct
            for vz in range(self.size[2]):
                s = 0.0

                for ox, oy in ti.ndrange(it_size.x, it_size.y):
                    offset = tm.ivec2(ox, oy)
                    h_pos = it_start + offset

                    hidden_state = hidden_states[h_pos.x, h_pos.y]

                    s += self.weights[h_pos.x, h_pos.y, hidden_state, ox, oy, vz, vt]

                s /= count

                self.reconstruction[vx, vy, vz, vt] = ti.cast(tm.exp(s - 1), param_type)

                if s > max_activation:
                    max_activation = s
                    max_index = vz

            # Update, if not early stopped
            for vz in range(self.size[2]):
                is_target = float(vz == visible_state)
                usage_increment = int(is_target)

                modulation = float(max_index != visible_state)

                delta = lr * modulation * (is_target - self.reconstruction[vx, vy, vz, vt])

                for ox, oy in ti.ndrange(it_size.x, it_size.y):
                    offset = tm.ivec2(ox, oy)
                    h_pos = it_start + offset

                    hidden_state = hidden_states[h_pos.x, h_pos.y]

                    # Weight indices
                    indices = (h_pos.x, h_pos.y, hidden_state, ox, oy, vz, vt)

                    self.weights[indices] += ti.cast(delta * hidden_gates[h_pos.x, h_pos.y], param_type)
                    self.usages[indices] = ti.cast(tm.min(usage_max, self.usages[indices] + usage_increment), usage_type)

    def write_buffers(self, fd: io.IOBase):
        write_from_buffer(fd, self.weights)
        write_from_buffer(fd, self.usages)

@ti.data_oriented
class Encoder:
    @ti.kernel
    def activate(self):
        for hx, hy in ti.ndrange(self.hidden_size[0], self.hidden_size[1]):
            max_index = 0
            max_activation = limit_min

            for hz in range(self.hidden_size[2]):
                activation = self.activations[hx, hy, hz]

                if activation > max_activation:
                    max_activation = activation
                    max_index = hz

            self.hidden_states[hx, hy] = max_index

    @ti.kernel
    def update_gates(self):
        for hx, hy in ti.ndrange(self.hidden_size[0], self.hidden_size[1]):
            self.hidden_gates[hx, hy] = ti.cast(tm.exp(-self.hidden_gates[hx, hy] * self.gcurve), param_type)

    def __init__(self, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ EncoderVisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            self.activations = ti.field(param_type, shape=hidden_size)
            self.activations.fill(0)

            self.hidden_states = ti.field(state_type, shape=(hidden_size[0], hidden_size[1]))
            self.hidden_states.fill(0)

            self.hidden_gates = ti.field(param_type, shape=(hidden_size[0], hidden_size[1]))

            self.vls = []

            for vld in vlds:
                self.vls.append(EncoderVisibleLayer(hidden_size, vld))

            for vl in self.vls:
                vl.init_random()

            # Hyperparameters
            self.lr = 0.5
            self.gcurve = 0.02

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iii", fd.read(3 * np.dtype(np.int32).itemsize))
            
            self.activations = ti.field(param_type, shape=self.hidden_size)
            self.activations.fill(0)

            self.hidden_states = ti.field(state_type, shape=(self.hidden_size[0], self.hidden_size[1]))
            read_into_buffer(fd, self.hidden_states)
            
            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.hidden_gates = ti.field(param_type, shape=(self.hidden_size[0], self.hidden_size[1]))

            for i in range(num_visible_layers):
                vld = EncoderVisibleLayerDesc()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius, vld.importance = struct.unpack("if", fd.read(np.dtype(np.int32).itemsize + np.dtype(np.float32).itemsize))

                vl = EncoderVisibleLayer(self.hidden_size, vld)

                read_into_buffer(fd, vl.weights)
                read_into_buffer(fd, vl.usages)

                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

    def step(self, visible_states: [ ti.Field ], vt_start: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        # Clear
        self.activations.fill(0)

        # Accumulate for all visible layers
        for i, vl in enumerate(self.vls):
            vl.accum_activations(self.hidden_size, vt_start, visible_states[i], self.activations)

        self.activate()

        if learn_enabled:
            # Clear
            self.hidden_gates.fill(0)

            # Accumulate gates for all visible layers
            for vl in self.vls:
                vl.accum_gates(self.hidden_size, self.hidden_states, self.hidden_gates)

            self.update_gates()

            for i, vl in enumerate(self.vls):
                vl.learn(self.hidden_size, vt_start, self.hidden_states, visible_states[i], self.hidden_gates, self.lr)

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iii", *self.hidden_size))

        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vls)))

        for i in range(len(self.vls)):
            vl = self.vls[i]

            fd.write(struct.pack("iiiiif", *vl.size, vl.radius, vl.importance))

            vl.write_buffers(fd)

        fd.write(struct.pack("ff", self.lr, self.gcurve))


        
