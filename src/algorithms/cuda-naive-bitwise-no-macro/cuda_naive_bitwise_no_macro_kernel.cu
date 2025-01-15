#ifndef CUDA_NAIVE_KERNEL_BITWISE_NO_MACRO_CU
#define CUDA_NAIVE_KERNEL_BITWISE_NO_MACRO_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_bitwise_no_macro.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"

namespace algorithms {

using idx_t = std::int64_t;

template <typename col_type>
using BitGridData = BitGridOnCudaWitOriginalSizes<col_type, idx_t>;

using CELL_STATE = bool;
constexpr CELL_STATE DEAD = false;
constexpr CELL_STATE ALIVE = true;

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type>
__device__ __forceinline__ col_type load_bit_col(idx_t x, idx_t y, col_type* source, BitGridData<col_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return source[get_idx(x, y, data.x_size)];
}

template <typename col_type>
__device__ __forceinline__ CELL_STATE get_cell_state(idx_t x, idx_t y, BitGridData<col_type> data) {
    constexpr int BITS = sizeof(col_type) * 8;

    if (x < 0 || x >= data.x_size_original || y < 0 || y >= data.y_size_original)
        return DEAD;

    auto bit_col = load_bit_col(x, y / BITS, data.input, data);
    auto y_bit = y % BITS;

    return (bit_col >> y_bit) & 1 ? ALIVE : DEAD;
}

template <typename col_type>
__device__ __forceinline__ void set_cell_state(idx_t x, idx_t y, CELL_STATE state, BitGridData<col_type> data) {
    constexpr int BITS = sizeof(col_type) * 8;

    if (x < 0 || x >= data.x_size_original || y < 0 || y >= data.y_size_original)
        return;

    auto bit_col = load_bit_col(x, y / BITS, data.output, data);
    auto y_bit = y % BITS;

    col_type one = 1;

    if (state == ALIVE) {
        bit_col |= (one << y_bit);
    }
    else {
        bit_col &= ~(one << y_bit);
    }

    data.output[get_idx(x, y / BITS, data.x_size_original)] = bit_col;
}

template <typename col_type>
__global__ void game_of_live_kernel(BitGridData<col_type> data) {
    constexpr int BITS = sizeof(col_type) * 8;

    idx_t x = blockIdx.x * blockDim.x + threadIdx.x;

    idx_t y_bit_col = blockIdx.y * blockDim.y + threadIdx.y;
    idx_t y_start = y_bit_col * BITS;

    idx_t x_size = data.x_size_original;
    idx_t y_size = data.y_size_original;

    for (std::size_t y = y_start; y < y_start + BITS; y++) {
        if (x >= x_size || y >= y_size)
            return;

        idx_t live_neighbors = 0;
        
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0)
                    continue;

                idx_t nx = x + i;
                idx_t ny = y + j;

                if (nx >= 0 && nx < x_size && ny >= 0 && ny < y_size) {
                    auto neighbor_state = get_cell_state(nx, ny, data);
                    
                    if (neighbor_state == ALIVE) {
                        live_neighbors++;
                    }
                }
            }
        }

        auto cell_state = get_cell_state(x, y, data);

        if (cell_state == ALIVE) {
            if (live_neighbors == 2 || live_neighbors == 3) {
                set_cell_state(x, y, ALIVE, data);
            }
            else {
                set_cell_state(x, y, DEAD, data);
            }
        }
        else {
            if (live_neighbors == 3) {
                set_cell_state(x, y, ALIVE, data);
            }
            else {
                set_cell_state(x, y, DEAD, data);
            }
        }
    }
}
template <std::size_t Bits>
void GoLCudaNaiveBitwiseNoMacro<Bits>::run_kernel(size_type iterations) {
    dim3 block(16, 16);
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

        game_of_live_kernel<<<grid, block>>>(cuda_data);
    }
}

} // namespace algorithms

template class algorithms::GoLCudaNaiveBitwiseNoMacro<16>;
template class algorithms::GoLCudaNaiveBitwiseNoMacro<32>;
template class algorithms::GoLCudaNaiveBitwiseNoMacro<64>;

#endif // CUDA_NAIVE_KERNEL_CU