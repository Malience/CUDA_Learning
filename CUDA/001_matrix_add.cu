__global__ void matrix_add(const float* X, const float* Y, float* out, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int i = idy * Nx + idx;
    if (idx < Nx && idy < Ny) out[i] = X[i] + Y[i];
}