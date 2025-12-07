from cuda.core.experimental import LaunchConfig
from launch_config import MAX_THREADS, WARP_SIZE, _nearest_pow_2, calc_config

# Random tests
assert 1 == _nearest_pow_2(0)
assert 2 == _nearest_pow_2(1)
assert 32 == _nearest_pow_2(18)
assert 8 == _nearest_pow_2(6)
assert 64 == _nearest_pow_2(42)
assert 256 == _nearest_pow_2(200)

# Robust test
test_range = 100000

# Start with pow 2
power = 2

assert(1 == _nearest_pow_2(0))
for i in range(1, test_range):
    nearest = _nearest_pow_2(i)
    assert power == nearest
    if i == power: power = power << 1

print("Nearest power of two test passed!")

def _check_block(tup):
    ln = len(tup)
    assert ln > 0
    assert ln < 4
    
    for i in range(ln):
        n = tup[i]
        assert n > 0 and (n & (n - 1)) == 0, "Block value not a power of 2"

    total = tup[0]
    for i in range(1, ln):
        total = total * tup[i] 
    
    assert total % WARP_SIZE == 0, "Block is not divisible by WARP_SIZE"
    assert total <= MAX_THREADS, "Block has too many threads!"

def _test_calc_config(size):
    config = calc_config(size)
    grid, block = config.grid, config.block

    assert len(grid) == len(block)
    if type(size) == int:
        assert block[0] * grid[0] >= size, "The config is smaller than the input size"
    else:
        for i in range(len(size)):
            assert block[i] * grid[i] >= size[i], "The config is smaller than the input size"

    _check_block(block)


# Test singles
test_range = 100000

for i in range(test_range):
    _test_calc_config(i)

for i in range(test_range):
    _test_calc_config((i,))

print("Config singles test past!")

# Test doubles
test_range_x = 513
test_range_y = 513

for x in range(test_range_x):
    for y in range(test_range_y):
        _test_calc_config((x, y))

_test_calc_config((123123, 120))
_test_calc_config((3333, 4433))
_test_calc_config((6453, 8))
_test_calc_config((4322, 120))
_test_calc_config((1, 123123))

print("Config doubles test past!")

# Test doubles
test_range_x = 100
test_range_y = 100
test_range_z = 50

_test_calc_config((17, 17, 1))

for x in range(test_range_x):
    for y in range(test_range_y):
        for z in range(test_range_z):
            _test_calc_config((x, y, z))

_test_calc_config((123123, 120, 32))
_test_calc_config((33, 44, 1231))
_test_calc_config((645, 8, 1))
_test_calc_config((432, 120, 5123))
_test_calc_config((1, 2, 1231))

print("Config triples test past!")