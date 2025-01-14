#ifndef CUDA_NAIVE_KERNEL_BITWISE_CU
#define CUDA_NAIVE_KERNEL_BITWISE_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_bitwise.hpp"
#include <cuda_runtime.h>

namespace algorithms {

using idx_t = std::int64_t;

__device__ static __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type>
__device__ static __forceinline__ col_type load(idx_t x, idx_t y, BitGridOnCuda<col_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename col_type>
__global__ static void game_of_live_kernel(BitGridOnCuda<col_type> data) {
    const idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= data.x_size || y >= data.y_size)
        return;

    const col_type lt = load(x - 1, y - 1, data);
    const col_type ct = load(x, y - 1, data);
    const col_type rt = load(x + 1, y - 1, data);

    const col_type lc = load(x - 1, y, data);
    const col_type cc = load(x, y, data);
    const col_type rc = load(x + 1, y, data);

    const col_type lb = load(x - 1, y + 1, data);
    const col_type cb = load(x, y + 1, data);
    const col_type rb = load(x + 1, y + 1, data);

    data.output[get_idx(x, y, data.x_size)] =
        CudaBitwiseOps<col_type>::compute_center_col(lt, ct, rt, lc, cc, rc, lb, cb, rb);
}

template <std::size_t Bits>
void GoLCudaNaiveBitwise<Bits>::run_kernel(size_type iterations) { // Added template parameter
    dim3 block(16, 16);
    dim3 grid((cuda_data.x_size + block.x - 1) / block.x, (cuda_data.y_size + block.y - 1) / block.y);

    for (std::size_t i = 0; i < iterations; ++i) {
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        game_of_live_kernel<<<grid, block>>>(cuda_data);
    }
}

} // namespace algorithms

template class algorithms::GoLCudaNaiveBitwise<16>;
template class algorithms::GoLCudaNaiveBitwise<32>;
template class algorithms::GoLCudaNaiveBitwise<64>;

#endif // CUDA_NAIVE_KERNEL_CU
