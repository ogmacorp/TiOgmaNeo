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
class DecoderVisibleLayerDesc:
    size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
    radius: int = 2

@ti.data_oriented
class DecoderVisibleLayer:
    weights: ti.Field
    usages: ti.Field
    visible_states_prev: ti.Field
    visible_gates: ti.Field

    size: (int, int, int, int)
    radius: int
    h_to_v: tm.vec2
    v_to_h: tm.vec2
    reverse_radii: tm.ivec2

    def __init__(self, hidden_size: (int, int, int, int), desc: DecoderVisibleLayerDesc):
        self.size = desc.size
        self.radius = desc.radius

        diam = self.radius * 2 + 1

        self.weights = ti.field(param_type, shape=(hidden_size[0], hidden_size[1], hidden_size[2], hidden_size[3], diam, diam, self.size[2], self.size[3]))

        self.usages = ti.field(usage_type, shape=self.weights.shape)
        self.usages.fill(0)

        self.visible_states_prev = ti.field(state_type, shape=(self.size[0], self.size[1], self.size[3]))
        self.visible_states_prev.fill(0)

        self.visible_gates = ti.field(ti.f16, shape=(self.size[0], self.size[1], self.size[3]))

        self.h_to_v = tm.vec2(self.size[0] / hidden_size[0], self.size[1] / hidden_size[1])
        self.v_to_h = tm.vec2(hidden_size[0] / self.size[0], hidden_size[1] / self.size[1])
        self.reverse_radii = tm.ivec2(math.ceil(self.v_to_h.x * diam * 0.5), math.ceil(self.v_to_h.y * diam * 0.5))

    # Initialization
    @ti.kernel
    def init_random(self):
        for hx, hy, hz, ht, ox, oy, vz, vt in self.weights:
            self.weights[hx, hy, hz, ht, ox, oy, vz, vt] = ti.cast((ti.random(dtype=param_type) * 2.0 - 1.0) * 0.01, param_type)

    # Stepping
    @ti.kernel
    def accum_activations(self, hidden_size: tm.ivec4, vt_start: int, visible_states: ti.template(), activations: ti.template()):
        for hx, hy, hz, ht in ti.ndrange(hidden_size.x, hidden_size.y, hidden_size.z, hidden_size.w):
            h_pos = tm.ivec2(hx, hy)

            v_center = project(h_pos, self.h_to_v)
            
            offset_start = v_center - self.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.size.x, v_center.x + 1 + self.radius), tm.min(self.size.y, v_center.y + 1 + self.radius))

            it_size = it_end - it_start

            s = 0
            count = it_size.x * it_size.y

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vt in range(self.size[3]):
                    visible_state = visible_states[v_pos.x, v_pos.y, (vt_start + vt) % self.size[3]]

                    s += self.weights[hx, hy, hz, ht, ox, oy, visible_state, vt]

            activations[hx, hy, hz, ht] += s / count

    @ti.kernel
    def update_gates(self, hidden_size: tm.ivec4, vt_start: int):
        for vx, vy, vt in ti.ndrange(self.size[0], self.size[1], self.size[3]):
            visible_state = self.visible_states_prev[vx, vy, (vt_start + vt) % self.size[3]]

            v_pos = tm.ivec2(vx, vy)

            h_center = project(v_pos, self.v_to_h)
            
            offset_start = h_center - self.reverse_radii

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(hidden_size.x, h_center.x + 1 + self.reverse_radii.x), tm.min(hidden_size.y, v_center.y + 1 + vl.reverse_radii.y))

            it_size = it_end - it_start

            s = 0
            count = it_size.x * it_size.y * hidden_size.z * hidden_size.w

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                h_pos = it_start + offset

                for hz in range(hidden_size.z):
                    for ht in range(hidden_size.w):
                        s += vl.usages[h_pos.x, h_pos.y, hz, ht, ox, oy, visible_state, vt]

            self.visible_gates[vx, vy, vt] = tm.exp(-s / count * self.gcurve)

    @ti.kernel
    def learn(self, hidden_size: tm.ivec4, vt_start: int, ht_start: int, target_temporal_horizon: int, target_hidden_states: ti.template(), activations: ti.template()):
        for hx, hy, hz, ht in ti.ndrange(hidden_size.x, hidden_size.y, hidden_size.z, hidden_size.w):
            h_pos = tm.ivec2(hx, hy)

            v_center = project(h_pos, self.h_to_v)
            
            offset_start = v_center - self.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.size.x, v_center.x + 1 + self.radius), tm.min(self.size.y, v_center.y + 1 + self.radius))

            it_size = it_end - it_start

            target_hidden_state = target_hidden_states[hx, hy, (ht_start + ht) % target_temporal_horizon]

            is_target = float(hz == target_hidden_state)
            usage_increment = int(is_target)

            delta = self.lr * (is_target - activations[hx, hy, hz, ht])

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vt in range(self.size[3]):
                    visible_state = self.visible_states_prev[v_pos.x, v_pos.y, (vt_start + vt) % self.size[3]]

                    # Weight indices
                    indices = (hx, hy, hz, ht, ox, oy, visible_state, vt)

                    self.weights[indices] += delta * self.visible_gates[vx, vy, vt]
                    self.usages[indices] = tm.min(255, self.usages[indices] + usage_increment)

    def write_buffers(self, fd: io.IOBase):
        write_from_buffer(fd, self.weights)
        write_from_buffer(fd, self.usages)
        write_from_buffer(fd, self.visible_states_prev)

@ti.data_oriented
class Decoder:
    @ti.kernel
    def activate(self):
        for hx, hy, ht in ti.ndrange(self.hidden_size[0], self.hidden_size[1], self.hidden_size[3]):
            max_index = 0
            max_activation = limit_min

            for hz in range(self.hidden_size[2]):
                activation = self.activations[hx, hy, hz, ht]

                if activation > max_activation:
                    max_activation = activation
                    max_index = hz

            self.hidden_states[hx, hy, ht] = max_index

            # Softmax
            total = 0

            for hz in range(self.hidden_size[2]):
                self.activations[hx, hy, hz, ht] = tm.exp(self.activations[hx, hy, hz, ht] - max_activation)

                total += self.activations[hx, hy, hz, ht]

            total_inv = 1 / tm.max(limit_small, total)

            for hz in range(self.hidden_size[2]):
                self.activations[hx, hy, hz, ht] *= total_inv

    def __init__(self, hidden_size: (int, int, int, int) = (4, 4, 16, 1), vlds: [ DecoderVisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            self.activations = ti.field(param_type, shape=hidden_size)
            self.activations.fill(0)

            self.hidden_states = ti.field(state_type, shape=(hidden_size[0], hidden_size[1], hidden_size[3]))
            self.hidden_states.fill(0)

            self.vls = []

            for vld in vlds:
                self.vls.append(DecoderVisibleLayer(hidden_size, vld))

            for vl in self.vls:
                vl.init_random()

            # Parameters
            self.lr = 1.0
            self.gcurve = 0.02

        else: # Load from h5py group
            self.hidden_size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))

            self.activations = ti.field(param_type, shape=self.hidden_size)
            read_into_buffer(fd, self.activations)

            self.hidden_states = ti.field(state_type, shape=(self.hidden_size[0], self.hidden_size[1], self.hidden_size[3]))
            read_into_buffer(fd, self.hidden_states)

            num_visible_layers = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

            self.vls = []

            for i in range(num_visible_layers):
                vld = DecoderVisibleLayerDesc()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

                vl = DecoderVisibleLayer(self.hidden_size, vld)

                read_into_buffer(fd, vl.weights)
                read_into_buffer(fd, vl.usages)
                read_into_buffer(fd, vl.visible_states_prev)

                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

    def step(self, visible_states: [ ti.Field ], target_hidden_states: ti.Field, vt_start: int, ht_start: int, target_temporal_horizon: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        if learn_enabled:
            # Activate gates
            for vl in self.vls:
                vl.update_gates(self.hidden_size, vt_start)

            for vl in self.vls:
                vl.learn(self.hidden_size, vt_start, ht_start, target_temporal_horizon, target_hidden_states, self.activations)

        # Clear
        self.activations.fill(0)

        # Accumulate for all visible layers
        for i, vl in enumerate(self.vls):
            vl.accum_activations(self.hidden_size, vt_start, visible_states[i], self.activations)

        self.activate()

        # Copy to prevs
        for i in range(len(self.vls)):
            vl = self.vls[i]

            vl.visible_states_prev.copy_from(visible_states[i])

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iiii", *self.hidden_size))

        write_from_buffer(fd, self.activations)
        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vls)))

        for i in range(len(self.vls)):
            vl = self.vls[i]

            fd.write(struct.pack("iiiii", *vl.size, vl.radius))

            vl.write_buffers(fd)

        fd.write(struct.pack("ff", self.lr, self.gcurve))
