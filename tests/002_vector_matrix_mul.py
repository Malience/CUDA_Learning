
import torch
from launch_config import launch_config, calc_config
from cupy_cuda import CupyCUDA

device = torch.device("cuda")
torch.set_default_device(device)

cc = CupyCUDA()
cc.compile_file("CUDA/002_vector_matrix_mul.cu", ("vector_matrix_mul",))

dtype = torch.float32
torch.set_default_dtype(dtype)

rng = torch.randint(100, 200, (2,))
size = tuple(rng.tolist())

v = torch.rand((size[1],))
m = torch.rand(size)

out = torch.empty((size[0],))

grid, blocks = calc_config(size[0])
args = (v.data_ptr(), m.data_ptr(), out.data_ptr(), *size)

cc.launch("vector_matrix_mul", grid, blocks, args)

# check result
assert torch.allclose(out, v @ m.T)

print(v)
print(m)
print(out)

print("Test passed successfully!")