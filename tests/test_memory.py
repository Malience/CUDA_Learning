
import torch
import launch_config
from cupy_cuda import CupyCUDA

import cupy as cp

# size of the vectors
size = 2048
BLOCKS = 2

# CUDA code with __constant__ memory
cuda_code = r'''
extern "C" {
#define BLOCKS 2

__constant__ float factors[BLOCKS];

__global__ void sum_and_multiply(const float * A, const float * B, float * C, const int size)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (item < size) {
        C[item] = (A[item] + B[item]) * factors[blockIdx.x];
    }
}
}
'''

cc = CupyCUDA()
cc.compile(cuda_code, ("sum_and_multiply",))

factors_ptr = cc.get_global("factors") 
fac = cp.ndarray(2, cp.float32, factors_ptr)
fac[...] = cp.random.random(2, dtype=cp.float32)

size = 2048
# Allocate and populate vectors on GPU (PyTorch tensors)
a_gpu = torch.rand(size, dtype=torch.float32, device="cuda")
b_gpu = torch.rand(size, dtype=torch.float32, device="cuda")
c_gpu = torch.zeros(size, dtype=torch.float32, device="cuda")
args = (a_gpu.data_ptr(), b_gpu.data_ptr(), c_gpu.data_ptr(), size)
6
cc.launch("sum_and_multiply", (2, 1, 1), (size // 2, 1, 1), args)


torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)

cuda_code = '''
#define NUM_PER_THREAD 32
#define NUMBERS 256
#define CONTANTS 256

__constant__ int constant_values[CONTANTS];

__global__ void histogram(const int* input, int* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int counts[NUMBERS];

    // Initialize counts
    if(threadIdx.x < NUMBERS) {
        counts[threadIdx.x] = 0;
    }
    __syncthreads();

    // Add numbers from input
    for(int i = 0; i < NUM_PER_THREAD; i++) {
        int idx2 = idx * NUM_PER_THREAD + i;
        int val = input[idx2];
        atomicAdd(&(counts[val]), 1);
    }
    __syncthreads();

    // Add numbers from constants
    if(threadIdx.x < CONTANTS) {
        int val = constant_values[threadIdx.x];
        atomicAdd(&(counts[val]), 1);
    }
    __syncthreads();

    
    if(threadIdx.x < NUMBERS) {
        atomicAdd(&(output[threadIdx.x]), counts[threadIdx.x]);
    }
}
'''

grid = 1
block = 512

NUM_PER_THREAD = 32
NUMBERS = 256
CONTANTS = 256

size = block * NUM_PER_THREAD

cc = CupyCUDA()
cc.compile(cuda_code, ("histogram",))

nums = torch.randint(0, NUMBERS, (size,), dtype=torch.int32)
out = torch.empty((NUMBERS,), dtype=torch.int32)
const = torch.randint(0, NUMBERS, (CONTANTS,), dtype=torch.int32)

cc.set_global("constant_values", const)

args = (nums.data_ptr(), out.data_ptr())

shared_mem = NUMBERS * torch.int32.itemsize

cc.launch("histogram", grid, block, args, shared_mem)

test = [0] * NUMBERS
for i in nums: test[i] = test[i] + 1
for i in const: test[i] = test[i] + 1
test = torch.tensor(test, dtype=torch.int32)

print(nums)
print(const)
print(out)
print(test)

assert torch.allclose(test, out)
print("Test passed successfully!")