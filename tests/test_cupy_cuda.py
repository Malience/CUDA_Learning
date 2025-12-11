import cupy as cp
import torch

from cupy_cuda import CupyCUDA

print(cp.cuda.get_current_stream())

dtype = torch.float32
torch.set_default_device("cuda")
torch.set_default_dtype(dtype)

cc = CupyCUDA()

print(cp.cuda.get_current_stream())


cc.compile_file("CUDA/000_vector_add.cu", ("vector_add",))

size = 1024

a = torch.rand(size)
b = torch.rand(size)
c = torch.empty_like(a)

cc.launch("vector_add", (1, 1, 1), (1024, 1, 1), a.data_ptr(), b.data_ptr(), c.data_ptr(), size)

print(a)
print(b)
print(c)

assert torch.allclose(a + b, c)
print("Test passed successfully!")