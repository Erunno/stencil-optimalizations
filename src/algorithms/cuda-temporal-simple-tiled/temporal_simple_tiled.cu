#ifndef CUDA_NAIVE_KERNEL_BITWISE_CU
#define CUDA_NAIVE_KERNEL_BITWISE_CU

#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"
#include "./temporal_simple_tiled.hpp"

namespace algorithms {

using idx_t = std::int64_t;
constexpr int block_x_size = 32;

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, TiledGridOnCuda<word_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename word_type, int block_x_size, int block_y_size>
void load_to_shared(idx_t x, idx_t y, TiledGridOnCuda<word_type> data, word_type* shared) {
    idx_t shared_x = threadIdx.x;
    idx_t shared_y = threadIdx.y;

    idx_t shared_idx = shared_y * block_x_size + shared_x;
    idx_t global_idx = get_idx(x, y, data.x_size);

    shared[shared_idx] = load(x, y, data);
}

template <typename bit_grid_mode, int block_y_size, int temporal_steps, typename word_type>
__global__ void game_of_live_kernel(TiledGridOnCuda<word_type> data) {
    int constexpr effective_x_block_size = block_x_size - 2;
    int constexpr effective_y_block_size = block_y_size - 2;

    idx_t x = blockIdx.x * effective_x_block_size + threadIdx.x - 1;
    idx_t y = blockIdx.y * effective_y_block_size + threadIdx.y - 1;

    idx_t x_to_shared = threadIdx.x + 1;
    idx_t y_to_shared = threadIdx.y + 1;

    int constexpr shared_x_size = block_x_size + 2;
    int constexpr shared_y_size = block_y_size + 2;

    __shared__ word_type shared[shared_x_size * shared_y_size];

    auto shm_idx = [=](idx_t x, idx_t y) {
        return y * shared_x_size + x;
    };

    shared[shm_idx(x_to_shared, y_to_shared)] = load(x, y, data);

    __syncthreads();

    for (int time_step = 0; time_step < temporal_steps; ++time_step) {
        word_type lt = shared[shm_idx(x_to_shared - 1, y_to_shared - 1)];
        word_type ct = shared[shm_idx(x_to_shared, y_to_shared - 1)];
        word_type rt = shared[shm_idx(x_to_shared + 1, y_to_shared - 1)];

        word_type lc = shared[shm_idx(x_to_shared - 1, y_to_shared)];
        word_type cc = shared[shm_idx(x_to_shared, y_to_shared)];
        word_type rc = shared[shm_idx(x_to_shared + 1, y_to_shared)];

        word_type lb = shared[shm_idx(x_to_shared - 1, y_to_shared + 1)];
        word_type cb = shared[shm_idx(x_to_shared, y_to_shared + 1)];
        word_type rb = shared[shm_idx(x_to_shared + 1, y_to_shared + 1)];

        word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);

        __syncthreads();
        
        shared[shm_idx(x_to_shared, y_to_shared)] = new_value;

        __syncthreads();
    }

    if (threadIdx.x == 0 || threadIdx.x == block_x_size - 1 || threadIdx.y == 0 || threadIdx.y == block_y_size - 1)
        return;

    data.output[get_idx(x, y, data.x_size)] = shared[shm_idx(x_to_shared, y_to_shared)];
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
template <int block_y_size, int temporal_steps>
void GoLCudaTemporalSimpleTiled<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) {
    dim3 block = {block_x_size, block_y_size};
    auto effective_x_block_size = block_x_size - 2;
    auto effective_y_block_size = block_y_size - 2;

    if (cuda_data.x_size % effective_x_block_size != 0 || cuda_data.y_size % effective_y_block_size != 0) {
        throw std::runtime_error("Grid size must be divisible by block size");
    }

    dim3 grid(cuda_data.x_size / effective_x_block_size, cuda_data.y_size / effective_y_block_size);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    auto big_steps = iterations / temporal_steps;

    for (std::size_t i = 0; i < big_steps; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i * temporal_steps;
            return;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<bit_grid_mode, block_y_size, temporal_steps><<<grid, block>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::GoLCudaTemporalSimpleTiled<common::CHAR, 64, algorithms::BitTileMode>;
template class algorithms::GoLCudaTemporalSimpleTiled<common::INT, 64, algorithms::BitTileMode>;

#endif // CUDA_NAIVE_KERNEL_CU