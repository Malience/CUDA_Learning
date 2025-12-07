
import torch
import launch_config
from torch_cuda import TorchCUDA

tc = TorchCUDA()
tc.compile_file("CUDA/000_vector_add.cu", ("vector_add",))

dtype = torch.float32

# prepare input
size = 128
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")

# prepare output
out = torch.empty_like(x)
# prepare launch
block = 64
grid = int((size + block - 1) // block)
print(f"Original grid-{grid}, block-{block}")

config = launch_config.calc_config(size)
print(f"Calculated grid-{config.grid}, block-{config.block}")
# config = launch_config.launch_config(grid=grid, block=block)
ker_args = (x.data_ptr(), y.data_ptr(), out.data_ptr(), size)
# launch kernel on PyTorch's stream
tc.launch("vector_add", config, *ker_args)

# check result
assert torch.allclose(out, x + y)
print("Test passed successfully!")