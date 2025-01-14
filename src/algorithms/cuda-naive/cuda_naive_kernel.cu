#ifndef CUDA_NAIVE_KERNEL_CU
#define CUDA_NAIVE_KERNEL_CU

#include "gol_cuda_naive.hpp"
#include "models.hpp"
#include <cuda_runtime.h>

namespace algorithms {

using idx_t = std::int64_t;

__device__ static __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

__global__ static void game_of_live_kernel(NaiveGridOnCuda data) {
    const idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    const idx_t x_size = data.x_size;
    const idx_t y_size = data.y_size;

    if (x >= x_size || y >= y_size)
        return;

    const idx_t idx = get_idx(x, y, x_size);

    idx_t live_neighbors = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0)
                continue;

            const idx_t nx = x + i;
            const idx_t ny = y + j;

            if (nx >= 0 && nx < x_size && ny >= 0 && ny < data.y_size) {
                live_neighbors += data.input[get_idx(nx, ny, x_size)];
            }
        }
    }

    if (data.input[idx] == 1) {
        data.output[idx] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
    }
    else {
        data.output[idx] = (live_neighbors == 3) ? 1 : 0;
    }
}

void GoLCudaNaive::run_kernel(size_type iterations) {
    const dim3 block(16, 16);
    const dim3 grid((cuda_data.x_size + block.x - 1) / block.x, (cuda_data.y_size + block.y - 1) / block.y);

    for (std::size_t i = 0; i < iterations; ++i) {
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<<<grid, block>>>(cuda_data);
    }
}

} // namespace algorithms

#endif // CUDA_NAIVE_KERNEL_CU
