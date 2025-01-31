#ifndef CUDA_NAIVE_KERNEL_BITWISE_CU
#define CUDA_NAIVE_KERNEL_BITWISE_CU

#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_bitwise.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"

namespace algorithms {

using idx_t = std::int64_t;

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, BitGridOnCuda<word_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename bit_grid_mode, typename word_type>
__global__ void game_of_live_kernel(BitGridOnCuda<word_type> data) {
    idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= data.x_size || y >= data.y_size)
        return;

    word_type lt = load(x - 1, y - 1, data);
    word_type ct = load(x, y - 1, data);
    word_type rt = load(x + 1, y - 1, data);

    word_type lc = load(x - 1, y, data);
    word_type cc = load(x, y, data);
    word_type rc = load(x + 1, y, data);

    word_type lb = load(x - 1, y + 1, data);
    word_type cb = load(x, y + 1, data);
    word_type rb = load(x + 1, y + 1, data);

    data.output[get_idx(x, y, data.x_size)] =
        CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
void GoLCudaNaiveBitwise<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) { // Added template parameter
    dim3 block = get_2d_block(this->params.thread_block_size);
    dim3 grid((cuda_data.x_size + block.x - 1) / block.x, (cuda_data.y_size + block.y - 1) / block.y);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            return;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<bit_grid_mode><<<grid, block>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 16, algorithms::BitColumnsMode>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 32, algorithms::BitColumnsMode>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 64, algorithms::BitColumnsMode>;

template class algorithms::GoLCudaNaiveBitwise<common::INT, 16, algorithms::BitColumnsMode>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 32, algorithms::BitColumnsMode>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 64, algorithms::BitColumnsMode>;

template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 16, algorithms::BitTileMode>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 32, algorithms::BitTileMode>;
template class algorithms::GoLCudaNaiveBitwise<common::CHAR, 64, algorithms::BitTileMode>;

template class algorithms::GoLCudaNaiveBitwise<common::INT, 16, algorithms::BitTileMode>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 32, algorithms::BitTileMode>;
template class algorithms::GoLCudaNaiveBitwise<common::INT, 64, algorithms::BitTileMode>;

#endif // CUDA_NAIVE_KERNEL_CU