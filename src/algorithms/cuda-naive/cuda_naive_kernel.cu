#ifndef CUDA_NAIVE_KERNEL_CU
#define CUDA_NAIVE_KERNEL_CU

#include "gol_cuda_naive.hpp"
#include "models.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"
#include "../_shared/cuda-helpers/border_policies.cuh"

namespace algorithms {

using idx_t = std::int64_t;
using border_policy_independent = border_policies::border_policy_independent;

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename border_policy, typename grid_cell_t>
__global__ void game_of_live_kernel(NaiveGridOnCuda<grid_cell_t> data) {
    idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    idx_t x_size = data.x_size;
    idx_t y_size = data.y_size;

    if (x >= x_size || y >= y_size)
        return;

    idx_t idx = border_policy_independent::get_idx(x, y, x_size);
    idx_t live_neighbors = 0;

    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            if (i == 0 && j == 0)
                continue;

            idx_t nx = x + i;
            idx_t ny = y + j;

            live_neighbors += border_policy::load(nx, ny, data);
        }
    }

    if (data.input[idx] == 1) {
        data.output[idx] = (live_neighbors == 2 || live_neighbors == 3) ? 1 : 0;
    }
    else {
        data.output[idx] = (live_neighbors == 3) ? 1 : 0;
    }
}

} // namespace

template <typename grid_cell_t>
template <typename border_policy>
void GoLCudaNaive<grid_cell_t>::run_kernel(size_type iterations) {
    dim3 block = get_2d_block(this->params.thread_block_size);
    dim3 grid((cuda_data.x_size + block.x - 1) / block.x, (cuda_data.y_size + block.y - 1) / block.y);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<border_policy><<<grid, block>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::GoLCudaNaive<common::CHAR>;
template class algorithms::GoLCudaNaive<common::INT>;

#endif // CUDA_NAIVE_KERNEL_CU