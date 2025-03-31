#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"
#include "./simple_warp_exchange.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdint>

namespace algorithms {

using idx_t = std::int64_t;
constexpr int x_block_size = 32;

namespace {

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, BitGridForSimpleWarpExchange<word_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename bit_grid_mode, typename word_type>
__global__ void game_of_live_kernel(BitGridForSimpleWarpExchange<word_type> data) {
    int constexpr effective_x_block_size = x_block_size - 2;

    idx_t x = blockIdx.x * effective_x_block_size + threadIdx.x - 1;
    idx_t y = blockIdx.y * blockDim.y + threadIdx.y;

    word_type lt = load(x - 1, y - 1, data);
    word_type ct = load(x, y - 1, data);
    word_type rt = load(x + 1, y - 1, data);

    word_type lc = load(x - 1, y, data);
    word_type cc = load(x, y, data);
    word_type rc = load(x + 1, y, data);

    word_type lb = load(x - 1, y + 1, data);
    word_type cb = load(x, y + 1, data);
    word_type rb = load(x + 1, y + 1, data);

    word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);

    if (threadIdx.x == 0 || threadIdx.x == x_block_size - 1)
        return;


    data.output[get_idx(x, y, data.x_size)] = new_value;
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
void GoLCudaSimpleWarpExchange<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) {
    dim3 block = {x_block_size, static_cast<std::uint32_t>(this->y_block_size)};
    auto effective_x_block_size = x_block_size - 2;
    auto effective_y_block_size = this->y_block_size;

    if (cuda_data.x_size % effective_x_block_size != 0 || cuda_data.y_size % effective_y_block_size != 0) {
        std::cerr << "Grid size: " << cuda_data.x_size << "x" << cuda_data.y_size << std::endl;
        std::cerr << "Effective block size: " << effective_x_block_size << "x" << effective_y_block_size << std::endl;
        throw std::runtime_error("Grid size must be divisible by block size");
    }

    dim3 grid(cuda_data.x_size / effective_x_block_size, cuda_data.y_size / effective_y_block_size);

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

template class algorithms::GoLCudaSimpleWarpExchange<common::CHAR, 32, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaSimpleWarpExchange<common::INT, 32, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaSimpleWarpExchange<common::CHAR, 64, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaSimpleWarpExchange<common::INT, 64, algorithms::WarpExchangeFullAdderOnRowsMode>;
