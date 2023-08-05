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
from .helpers import *

@ti.data_oriented
class Encoder:
    @ti.dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
        radius: int = 2
        importance: float = 1.0

    class VisibleLayer:
        weights: ti.Field
        usages: ti.Field
        reconstruction: ti.Field

        h_to_v: tm.vec2
        v_to_h: tm.vec2
        reverse_radii: tm.ivec2

    # Initialization
    @ti.kernel
    def init(i: int):
        vl = self.vls[i]

        for hx, hy, hz, ox, oy, vz, vt in weights:
            vl.weights[hx, hy, hz, ox, oy, vz, vt] = ti.random()

    # Stepping
    @ti.kernel
    def accum_activations(i: int, vt_start: int, visible_states: ti.Field):
        vld = self.vlds[i]
        vl = self.vls[i]

        for hx, hy, hz in ti.ndrange(self.hidden_size):
            h_pos = tm.ivec2(hx, hy)

            v_center = project(h_pos, vld.h_to_v)
            
            offset_start = v_center - vld.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(vld.size.x, v_center.x + 1 + vld.radius), tm.min(vld.size.y, v_center.y + 1 + vld.radius))

            it_size = it_end - it_start

            s = 0
            count = it_size.x * it_size.y

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vt in range(vld.size[3]):
                    visible_state = visible_states[v_pos.x, v_pos.y, (vt_start + vt) % vld.size[3]]

                    s += vl.weights[hx, hy, hz, ox, oy, visible_state, vt]

            self.activations[hx, hy, hz] += s / count * vld.importance

    @ti.kernel
    def activate():
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
    def accum_gates(i: int):
        vld = self.vlds[i]
        vl = self.vls[i]

        for hx, hy in ti.ndrange(self.hidden_size[0], self.hidden_size[1]):
            h_pos = tm.ivec2(hx, hy)

            v_center = project(h_pos, vld.h_to_v)
            
            offset_start = v_center - vld.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(vld.size.x, v_center.x + 1 + vld.radius), tm.min(vld.size.y, v_center.y + 1 + vld.radius))

            it_size = it_end - it_start

            hidden_state = self.hidden_states[hx, hy]

            s = 0
            count = it_size.x * it_size.y * vld.size[2] * vld.size[3]

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vz in range(vld.size[2]):
                    for vt in range(vld.size[3]):
                        s += vl.usages[hx, hy, hidden_state, ox, oy, vz, vt]

            self.hidden_gates[hx, hy] += s / count

    @ti.kernel
    def update_gates():
        for hx, hy in ti.ndrange(self.hidden_size[0], self.hidden_size[1]):
            self.hidden_gates[hx, hy] = tm.exp(-self.hidden_gates[hx, hy] / len(self.vls) * self.gcurve)

    @ti.kernel
    def learn(i: int, vt_start: int, visible_states: ti.Field):
        vld = self.vlds[i]
        vl = self.vls[i]

        for vx, vy, vt in ti.ndrange(vld.size[0], vld.size[1], vld.size[3]):
            visible_state = visible_states[vx, vy, (vt_start + vt) % vld.size[3]]

            v_pos = tm.ivec2(vx, vy)

            h_center = project(v_pos, vld.v_to_h)
            
            offset_start = h_center - vl.reverse_radii

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.hidden_size[0], h_center.x + 1 + vl.reverse_radii.x), tm.min(self.hidden_size[1], v_center.y + 1 + vl.reverse_radii.y))

            it_size = it_end - it_start

            count = it_size.x * it_size.y

            max_index = 0
            max_activation = limit_min

            # Reconstruct
            for vz in range(vld.size[2]):
                s = 0

                for ox, oy in ti.ndrange(it_size):
                    offset = tm.ivec2(ox, oy)
                    h_pos = it_start + offset

                    hidden_state = self.hidden_states[h_pos.x, h_pos.y]

                    s += vl.weights[h_pos.x, h_pos.y, hidden_state, ox, oy, vz, vt]

                s /= count

                vl.reconstruction[vx, vy, vz, vt] = tm.exp(s - 1)

                if s > max_activation:
                    max_activation = s
                    max_index = vz

            # Update, if not early stopped
            for vz in range(vld.size[2]):
                is_target = float(vz == visible_state)
                usage_increment = int(is_target)

                modulation = float(max_index != visible_state)

                delta = self.lr * modulation * (is_target - vl.reconstruction[vx, vy, vz, vt])

                for ox, oy in ti.ndrange(it_size):
                    offset = tm.ivec2(ox, oy)
                    h_pos = it_start + offset

                    hidden_state = self.hidden_states[h_pos.x, h_pos.y]

                    # Weight indices
                    indices = (h_pos.x, h_pos.y, hidden_state, ox, oy, vz, vt)

                    vl.weights[indices] += delta * self.hidden_gates[h_pos.x, h_pos.y]
                    vl.usages[indices] = tm.min(255, vl.usages[indices] + usage_increment)

    def __init__(self, hidden_size: (int, int, int) = (4, 4, 16), vlds: [ VisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            self.activations = ti.field(param_type, shape=hidden_size)
            self.activations.fill(0)

            self.hidden_states = ti.field(state_type, shape=(hidden_size[0], hidden_size[1]))
            self.hidden_states.fill(0)

            self.hidden_gates = ti.field(param_type, shape=(hidden_size[0], hidden_size[1]))

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                diam = vld.radius * 2 + 1

                vl.weights = ti.field(param_type, shape=(hidden_size[0], hidden_size[1], hidden_size[2], diam, diam, vld.size[2], vld.size[3]))

                vl.usages = ti.field(usage_type, shape=vl.weights.shape)
                vl.usages.fill(0)

                vl.reconstruction = ti.field(param_type, shape=vld.size)

                vl.h_to_v = tm.vec2(vld.size[0] / hidden_size[0], vld.size[1] / hidden_size[1])
                vl.v_to_h = tm.vec2(hidden_size[0] / vld.size[0], hidden_size[1] / vld.size[1])
                vl.reverse_radii = tm.ivec2(math.ceil(vl.v_to_h.x * diam * 0.5), math.ceilf(vl.v_to_h.y * diam * 0.5))

                self.init(i)

                self.vls.append(vl)

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

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius, vld.importance = struct.unpack("if", fd.read(np.dtype(np.int32).itemsize + np.dtype(np.float32).itemsize))

                num_visible_columns = vld.size[0] * vld.size[1] * vld.size[3]
                num_visible_cells = num_visible_columns * vld.size[2]

                vl.weights = ti.field(param_type, shape=(self.hidden_size[0], self.hidden_size[1], self.hidden_size[2], diam, diam, vld.size[2], vld.size[3]))
                read_into_buffer(fd, vl.weights)

                vl.usages = ti.field(usage_type, shape=vl.weights.shape)
                read_into_buffer(fd, vl.usages)

                vl.reconstruction = ti.field(param_type, shape=vld.size)

                vl.h_to_v = tm.vec2(vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1])
                vl.v_to_h = tm.vec2(self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1])
                vl.reverse_radii = tm.ivec2(math.ceil(vl.v_to_h.x * diam * 0.5), math.ceilf(vl.v_to_h.y * diam * 0.5))

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

    def step(self, visible_states: [ ti.Field ], vt_start: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        # Clear
        self.activations.fill(0)

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            self.accum_activations(i, vt_start, visible_states[i])

        self.activate()

        if learn_enabled:
            # Clear
            self.hidden_gates.fill(0)

            # Accumulate gates for all visible layers
            for i in range(len(self.vls)):
                self.accum_gates(i)

            self.update_gates()

            for i in range(len(self.vls)):
                self.learn(i, vt_start, visible_states[i])

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iii", *self.hidden_size))

        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiiiif", *vld.size, vld.radius, vld.importance))

            write_from_buffer(fd, vl.weights)
            write_from_buffer(fd, vl.usages)

        fd.write(struct.pack("ff", self.lr, self.gcurve))


        
