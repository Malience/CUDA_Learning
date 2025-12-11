from typing import Tuple, Union
from cuda.core.experimental import LaunchConfig


MAX_THREADS = 512 # 1024
# MAX_BLOCK_SIZE = (1024, 1024, 64)
WARP_SIZE = 32


def launch_config(grid: Union[Tuple, int] = None, block: Union[Tuple, int] = None, shared_memory: int = 0) -> LaunchConfig:
    return LaunchConfig(grid=grid, block=block, shmem_size=shared_memory)

def _nearest_pow_2(value: int) -> int:
    if value == 0: return 1
    if value == 1: return 2
    if (value & (value - 1)) == 0: return value
    return 0b1 << value.bit_length()

def _calc_grid(total_size, block_size) -> int:
    g = (total_size + block_size - 1) // block_size
    return max(1, g)

def _calc_single(size: int) -> Tuple[int, int]:
    # It should always be at least as large as the warp size
    if size <= WARP_SIZE: return 1, WARP_SIZE

    # It cannot exceed the maximum number of threads available on the hardware
    # Optimally, this is 512
    # Additionally, the block_value should always be a multiple of the warp size
    if size < MAX_THREADS:
        return 1, _nearest_pow_2(size)
    
    # Otherwise the block size is set to the maximum and we calculate the grid
    return _calc_grid(size, MAX_THREADS), MAX_THREADS

def _calc_double(x: int, y: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if x < 1: x = 1
    if y < 1: y = 1

    wx, wy = _nearest_pow_2(x), _nearest_pow_2(y)

    if wx * wy <= MAX_THREADS: # Fit y and pump up x
        if wx * wy < WARP_SIZE:
            if wx >= wy:
                wx = WARP_SIZE // wy # Set the total size to 32
            else:
                wy = WARP_SIZE // wx
        return (1, 1), (wx, wy)
    
    #TODO: Update for an algorithm that focuses more on occupancy
    # Slowly step down each until they are within bounds
    while wx * wy > MAX_THREADS:
        if wx >= wy: wx = wx >> 1
        else: wy = wy >> 1

    gx, gy = _calc_grid(x, wx), _calc_grid(y, wy)
    return (gx, gy), (wx, wy)

def _largest_triple(x: int, y: int, z: int) -> int:
    if x >= y and x >= z: return 0
    if y >= x and y >= z: return 1
    return 2

def _calc_triple(x: int, y: int, z: int) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    if x < 1: x = 1
    if y < 1: y = 1
    if z < 1: z = 1

    wx, wy, wz = _nearest_pow_2(x), _nearest_pow_2(y), _nearest_pow_2(z)

    if wx * wy * wz <= MAX_THREADS: # Fit y and pump up x
        if wx * wy * wz < WARP_SIZE:
            # Set the total size to 32
            match _largest_triple(wx, wy, wz):
                case 0: wx = (WARP_SIZE // wy) // wz
                case 1: wy = (WARP_SIZE // wx) // wz
                case 2: wz = (WARP_SIZE // wy) // wx
        return (1, 1, 1), (wx, wy, wz)
    
    #TODO: Update for an algorithm that focuses more on occupancy
    # Slowly step down each until they are within bounds
    while wx * wy * wz > MAX_THREADS:
        match _largest_triple(wx, wy, wz):
            case 0: wx = wx >> 1
            case 1: wy = wy >> 1
            case 2: wz = wz >> 1
    
    gx, gy, gz = _calc_grid(x, wx), _calc_grid(y, wy), _calc_grid(z, wz)
    return (gx, gy, gz), (wx, wy, wz)


def calc_config(size: Union[Tuple, int]):
    grid, block = None, None
    if type(size) == int:
        grid, block = _calc_single(size)
    elif type(size) == tuple:
        dim = len(size)
        if dim < 1 or dim > 3:
            raise Exception(f"ERROR: Invalid dimensions: {dim}")
        if dim == 1:
            grid, block = _calc_single(size[0])
        elif dim == 2:
            grid, block = _calc_double(size[0], size[1])
        else:
            grid, block = _calc_triple(size[0], size[1], size[2])
    else:
        raise Exception("ERROR: Invalid shape type, must be an int or a tuple")
    
    return grid, block

# def calc_config(size: Union[Tuple, int], shared_memory: int = 0) -> LaunchConfig:
#     grid, block = None, None
#     if type(size) == int:
#         grid, block = _calc_single(size)
#     elif type(size) == tuple:
#         dim = len(size)
#         if dim < 1 or dim > 3:
#             raise Exception(f"ERROR: Invalid dimensions: {dim}")
#         if dim == 1:
#             grid, block = _calc_single(size[0])
#         elif dim == 2:
#             grid, block = _calc_double(size[0], size[1])
#         else:
#             grid, block = _calc_triple(size[0], size[1], size[2])
#     else:
#         raise Exception("ERROR: Invalid shape type, must be an int or a tuple")
    
#     return LaunchConfig(grid=grid, block=block, shmem_size=shared_memory)
