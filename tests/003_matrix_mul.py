
import torch
from launch_config import launch_config, calc_config
from cupy_cuda import CupyCUDA


device = torch.device("cuda")
torch.set_default_device(device)

cc = CupyCUDA()
cc.compile_file("CUDA/003_matrix_mul.cu", ("matrix_mul",))

dtype = torch.float32
torch.set_default_dtype(dtype)

rng = torch.randint(20, 500, (3,))
dims = tuple(rng.tolist())

size_a = dims[:2]
size_b = dims[1:]
size_out = (size_a[0], size_b[1])

a = torch.rand(size_a)
b = torch.rand(size_b)

out = torch.empty(size_out)

grid, block = calc_config(size_out)
args = (a.data_ptr(), b.data_ptr(), out.data_ptr(), *dims)
cc.launch("matrix_mul", grid, block, args)

print(a)
print(b)
print(a @ b)
print(out)

assert torch.allclose(out, a @ b)
print("Test passed successfully!")