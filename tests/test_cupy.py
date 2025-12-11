import cupy
import numpy as np
import torch

# size of the vectors
size = 1024

dtype = cupy.float32

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=dtype)
b_gpu = cupy.random.rand(size, dtype=dtype)
c_gpu = cupy.zeros(size, dtype=dtype)

dtype = torch.float32

# So we can use torch
a = torch.rand(size, dtype=dtype, device="cuda")
b = torch.rand(size, dtype=dtype, device="cuda")
c = torch.empty_like(a)

# CUDA vector_add
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a.data_ptr(), b.data_ptr(), c.data_ptr(), size))


print(a)
print(b)
print(c)

assert torch.allclose(a + b, c)


code = None
with open("CUDA/000_vector_add.cu", 'r') as f:
    code = f.read()

a = torch.rand(size, dtype=dtype, device="cuda")
b = torch.rand(size, dtype=dtype, device="cuda")
c = torch.empty_like(a)

mod = cupy.RawModule(code=code, options=('--std=c++17',), name_expressions=("vector_add",))
vector_add_gpu = mod.get_function("vector_add")
vector_add_gpu((1, 1, 1), (size, 1, 1), (a.data_ptr(), b.data_ptr(), c.data_ptr(), size))


print(a)
print(b)
print(c)

assert torch.allclose(a + b, c)
print("Test passed successfully!")