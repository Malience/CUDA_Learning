# CUDALearning

Simple repository for learning CUDA programming. The ultimate goal is to implement, accelerate, and optimize advanced Reinforcement Learning algorithms into CUDA.

## Requirements

- CUDA-capable GPU and matching CUDA toolkit (e.g., CUDA 11.x)
- Python
- Recommended: virtual environment (venv / conda)
- PyTorch (Good luck 3.8 seems glitchy)

## Progress

### Session 1 ==> 000-003

- Implemented a basic python environment using cuda.base. I have implemented a simple object to compile and run CUDA code in python. (check torch_cuda.py)
- Added a robust semi-optimized grid+block calculator to allow for randomized data sizes. Can be somewhat inefficient with wierd inputs, but it prevents invalid values and gauruntees the block will be divisible by the usual GPU WARP size.
- With those I finished some simple linear algebra examples to brush the rust off.
