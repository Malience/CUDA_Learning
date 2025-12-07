// MAtrix multiplication between A (NxM) and B (MxP)
__global__ void matrix_mul(const float* A, const float* B, float* out, int N, int M, int P) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < N && idy < P) {
        float o = 0;
        for(int i = 0; i < M; i++){
            o += A[idx * M + i] * B[idy + i * P];
        }
        out[idx * P + idy] = o;
    }
}