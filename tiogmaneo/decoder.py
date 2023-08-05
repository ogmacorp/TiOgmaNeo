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
class Decoder:
    @ti.dataclass
    class VisibleLayerDesc:
        size: (int, int, int, int) = (4, 4, 16, 1) # Width, height, column size, temporal size
        radius: int = 2
        
    class VisibleLayer:
        weights: ti.Field
        usages: ti.Field
        visible_states_prev: ti.Field
        visible_gates: ti.Field

        h_to_v: tm.vec2
        v_to_h: tm.vec2
        reverse_radii: tm.ivec2

    # Initialization
    @ti.kernel
    def init(i: int):
        vl = self.vls[i]

        for hx, hy, hz, ht, ox, oy, vz, vt in weights:
            vl.weights[hx, hy, hz, ht, ox, oy, vz, vt] = ti.random()

    # Stepping
    @ti.kernel
    def accum(i: int, vt_start: int, visible_states: ti.Field):
        vld = self.vlds[i]
        vl = self.vls[i]

        for hx, hy, hz, ht in ti.ndrange(self.hidden_size):
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

                    s += vl.weights[hx, hy, hz, ht, ox, oy, visible_state, vt]

            self.activations[hx, hy, hz, ht] += s / count

    @ti.kernel
    def activate():
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

    @ti.kernel
    def update_gates(i: int, vt_start: int):
        vld = self.vlds[i]
        vl = self.vls[i]

        for vx, vy, vt in ti.ndrange(vld.size[0], vld.size[1], vld.size[3]):
            visible_state = vl.visible_states_prev[vx, vy, (vt_start + vt) % vld.size[3]]

            v_pos = tm.ivec2(vx, vy)

            h_center = project(v_pos, vld.v_to_h)
            
            offset_start = h_center - vl.reverse_radii

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(self.hidden_size[0], h_center.x + 1 + vl.reverse_radii.x), tm.min(self.hidden_size[1], v_center.y + 1 + vl.reverse_radii.y))

            it_size = it_end - it_start

            s = 0
            count = it_size.x * it_size.y * self.hidden_size[2]

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                h_pos = it_start + offset

                for hz in range(self.hidden_size[2]):
                    for ht in range(self.hidden_size[3]):
                        s += vl.usages[h_pos.x, h_pos.y, hz, ht, ox, oy, visible_state, vt]

            self.visible_gates[vx, vy, vt] = tm.exp(-s / count * self.gcurve)

    @ti.kernel
    def learn(i: int, vt_start: int, ht_start: int, target_temporal_horizon: int, target_hidden_states: ti.Field):
        vld = self.vlds[i]
        vl = self.vls[i]

        for hx, hy, hz, ht in ti.ndrange(self.hidden_size):
            h_pos = tm.ivec2(hx, hy)

            v_center = project(h_pos, vld.h_to_v)
            
            offset_start = v_center - vld.radius

            it_start = tm.ivec2(tm.max(0, offset_start.x), tm.max(0, offset_start.y))
            it_end = tm.ivec2(tm.min(vld.size.x, v_center.x + 1 + vld.radius), tm.min(vld.size.y, v_center.y + 1 + vld.radius))

            it_size = it_end - it_start

            target_hidden_state = target_hidden_states[hx, hy, (ht_start + ht) % target_temporal_horizon]

            is_target = float(hz == target_hidden_state)
            usage_increment = int(is_target)

            delta = self.lr * (is_target - self.activations[hx, hy, hz, ht])

            for ox, oy in ti.ndrange(it_size):
                offset = tm.ivec2(ox, oy)
                v_pos = it_start + offset

                for vt in range(vld.size[3]):
                    visible_state = self.visible_states_prev[v_pos.x, v_pos.y, (vt_start + vt) % vld.size[3]]

                    # Weight indices
                    indices = (hx, hy, hz, ht, ox, oy, visible_state, vt)

                    vl.weights[indices] += delta * vl.visible_gates[vx, vy, vt]
                    vl.usages[indices] = tm.min(255, vl.usages[indices] + usage_increment)

    def __init__(self, hidden_size: (int, int, int, int) = (4, 4, 16, 1), vlds: [ VisibleLayerDesc ] = [], fd: io.IOBase = None):
        if fd is None:
            self.hidden_size = hidden_size

            self.activations = ti.field(param_type, shape=hidden_size)
            self.activations.fill(0)

            self.hidden_states = ti.field(state_type, shape=(hidden_size[0], hidden_size[1], hidden_size[3]))
            self.hidden_states.fill(0)

            self.vlds = vlds
            self.vls = []

            for i in range(len(vlds)):
                vld = self.vlds[i]
                vl = self.VisibleLayer()

                diam = vld.radius * 2 + 1

                vl.weights = ti.field(param_type, shape=(hidden_size[0], hidden_size[1], hidden_size[2], hidden_size[3], diam, diam, vld.size[2], vld.size[3]))

                vl.usages = ti.field(usage_type, shape=vl.weights.shape)
                vl.usages.fill(0)

                vl.visible_states_prev = ti.field(state_type, shape=(vld.size[0], vld.size[1], vld.size[3]))
                vl.visible_states_prev.fill(0)

                vl.visible_gates = ti.field(ti.f16, shape=(vld.size[0], vld.size[1], vld.size[3]))

                vl.h_to_v = tm.vec2(vld.size[0] / hidden_size[0], vld.size[1] / hidden_size[1])
                vl.v_to_h = tm.vec2(hidden_size[0] / vld.size[0], hidden_size[1] / vld.size[1])
                vl.reverse_radii = tm.ivec2(math.ceil(vl.v_to_h.x * diam * 0.5), math.ceilf(vl.v_to_h.y * diam * 0.5))

                self.init(i)

                self.vls.append(vl)

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

            self.vlds = []
            self.vls = []

            for i in range(num_visible_layers):
                vld = self.VisibleLayerDesc()
                vl = self.VisibleLayer()

                vld.size = struct.unpack("iiii", fd.read(4 * np.dtype(np.int32).itemsize))
                vld.radius = struct.unpack("i", fd.read(np.dtype(np.int32).itemsize))[0]

                diam = vld.radius * 2 + 1

                vl.weights = ti.field(param_type, shape=(self.hidden_size[0], self.hidden_size[1], self.hidden_size[2], self.hidden_size[3], diam, diam, vld.size[2], vld.size[3]))
                read_into_buffer(fd, vl.weights)

                vl.usages = ti.field(usage_type, shape=vl.weights.shape)
                read_into_buffer(fd, vl.usages)

                vl.visible_states_prev = ti.field(state_type, shape=(vld.size[0], vld.size[1], vld.size[3]))
                read_into_buffer(fd, vl.visible_states_prev)

                vl.visible_gates = ti.field(ti.f16, shape=(vld.size[0], vld.size[1], vld.size[3]))

                vl.h_to_v = tm.vec2(vld.size[0] / self.hidden_size[0], vld.size[1] / self.hidden_size[1])
                vl.v_to_h = tm.vec2(self.hidden_size[0] / vld.size[0], self.hidden_size[1] / vld.size[1])
                vl.reverse_radii = tm.ivec2(math.ceil(vl.v_to_h.x * diam * 0.5), math.ceilf(vl.v_to_h.y * diam * 0.5))

                self.vlds.append(vld)
                self.vls.append(vl)

            # Parameters
            self.lr, self.gcurve = struct.unpack("ff", fd.read(2 * np.dtype(np.float32).itemsize))

    def step(self, visible_states: [ ti.Field ], target_hidden_states: ti.Field, vt_start: int, ht_start: int, target_temporal_horizon: int, learn_enabled: bool = True):
        assert(len(visible_states) == len(self.vls))

        if learn_enabled:
            # Activate gates
            for i in range(len(self.vls)):
                self.update_gates(i, vt_start)

            for i in range(len(self.vls)):
                self.learn(i, vt_start, ht_start, target_temporal_horizon, target_hidden_states)

        # Clear
        self.activations.fill(0)

        # Accumulate for all visible layers
        for i in range(len(self.vls)):
            self.accum(i, vt_start, visible_states[i])

        self.activate()

        # Copy to prevs
        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            vl.visible_states_prev = visible_states[i].copy()

    def write(self, fd: io.IOBase):
        fd.write(struct.pack("iiii", *self.hidden_size))

        write_from_buffer(fd, self.activations)
        write_from_buffer(fd, self.hidden_states)

        fd.write(struct.pack("i", len(self.vlds)))

        for i in range(len(self.vls)):
            vld = self.vlds[i]
            vl = self.vls[i]

            fd.write(struct.pack("iiiii", *vld.size, vld.radius))

            write_from_buffer(fd, vl.weights)
            write_from_buffer(fd, vl.usages)
            write_from_buffer(fd, vl.visible_states_prev)

        fd.write(struct.pack("ff", self.lr, self.gcurve))
