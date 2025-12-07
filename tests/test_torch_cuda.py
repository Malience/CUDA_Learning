import torch
import launch_config
from torch_cuda import TorchCUDA

code = """
template<typename T>
__global__ void saxpy_kernel(const T* a, const T* x, const T* y, T* out, size_t N) {
 const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
 if (tid < N) {
   // Dereference a to get the scalar value
   out[tid] = (*a) * x[tid] + y[tid];
 }
}
"""

tc = TorchCUDA()
tc.compile(code, ("saxpy_kernel<float>", "saxpy_kernel<double>"))

dtype = torch.float32

# prepare input/output
size = 64
# Use a single element tensor for 'a'
a = torch.tensor([10.0], dtype=dtype, device="cuda")
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")
out = torch.empty_like(x)

# prepare launch
block = 32
grid = int((size + block - 1) // block)

config = launch_config.launch_config(grid=grid, block=block)
ker_args = (a.data_ptr(), x.data_ptr(), y.data_ptr(), out.data_ptr(), size)
tc.launch("saxpy_kernel<float>", config, *ker_args)

# check result
assert torch.allclose(out, a.item() * x + y)
print("Single precision test passed!")

# let's repeat again with double precision
dtype = torch.float64

# prepare input
size = 128
# Use a single element tensor for 'a'
a = torch.tensor([42.0], dtype=dtype, device="cuda")
x = torch.rand(size, dtype=dtype, device="cuda")
y = torch.rand(size, dtype=dtype, device="cuda")

# prepare output
out = torch.empty_like(x)

# prepare launch
block = 64
grid = int((size + block - 1) // block)
config = launch_config.launch_config(grid=grid, block=block)
ker_args = (a.data_ptr(), x.data_ptr(), y.data_ptr(), out.data_ptr(), size)

# launch kernel on PyTorch's stream
tc.launch("saxpy_kernel<double>", config, *ker_args)

# check result
assert torch.allclose(out, a * x + y)
print("Double precision test passed!")
print("All tests passed successfully!")