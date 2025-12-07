__global__ void vector_matrix_mul(const float* V, const float* M, float* out, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < Nx) {
        float o = 0;
        for(int i = 0; i < Ny; i++){
            o += V[i] * M[idx * Ny + i];
        }
        out[idx] = o;
    }
}