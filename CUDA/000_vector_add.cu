    __global__ void vector_add(const float* X, const float* Y, float* out, int N) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < N) out[i] = X[i] + Y[i];
 }