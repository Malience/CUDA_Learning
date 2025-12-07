
import torch
import launch_config
from torch_cuda import TorchCUDA

tc = TorchCUDA()
tc.compile_file("CUDA/001_matrix_add.cu", ("matrix_add",))

dtype = torch.float32

rng = torch.randint(1, 5000, (2,))
size = tuple(rng.tolist())
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")

out = torch.empty_like(x)

# I just spent three hours creating a semi-optimal launch config calculator so I don't need this
# I'm so cool
# block = (32, 32)
# grid = (4, 4)
# config = launch_config.launch_config(grid=grid, block=block)

config = launch_config.calc_config(size)
args = (x.data_ptr(), y.data_ptr(), out.data_ptr(), *size)

tc.launch("matrix_add", config, *args)

# check result
assert torch.allclose(out, x + y)
print("Test passed successfully!")

print(x)
print(y)
print(out)