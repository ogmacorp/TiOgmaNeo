# ----------------------------------------------------------------------------
#  TiOgmaNeo
#  Copyright(c) 2023 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of TiOgmaNeo is licensed to you under the terms described
#  in the TIOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

import numpy as np
import io
import taichi as ti
import taichi.math as tm

# Types used for aspects of SPH
state_type = ti.i32
param_type = ti.f16
usage_type = ti.u8

# Limits
limit_min = -9999.0
limit_max = 9999.0
limit_small = 0.001

# Some useful helpers
@ti.func
def project(pos: tm.ivec2, scalars: tm.vec2):
    return tm.ivec2((pos.x + 0.5) * scalars.x, (pos.y + 0.5) * scalars.y)

@ti.func
def in_bounds0(pos: tm.ivec2, upper_bound: tm.ivec2):
    return pos.x >= 0 and pos.y >= 0 and pos.x < upper_bound.x and pos.y < upper_bound.y

@ti.func
def in_bounds(pos: tm.ivec2, lower_bound: tm.ivec2, upper_bound: tm.ivec2):
    return pos.x >= lower_bound.x and pos.y >= lower_bound.y and pos.x < upper_bound.x and pos.y < upper_bound.y

# Serialization/deserialization
def read_array(fd: io.IOBase, count: int, dtype: np.dtype):
    return np.frombuffer(fd.read(count * np.dtype(dtype).itemsize), dtype)

def read_into_buffer(fd: io.IOBase, buffer: ti.Field):
    buffer.from_numpy(read_array(fd, len(buffer), buffer.dtype))

def write_array(fd: io.IOBase, arr: np.array):
    fd.write(arr.tobytes())
    
def write_from_buffer(fd: io.IOBase, buffer: ti.Field):
    write_array(fd, buffer.to_numpy())
