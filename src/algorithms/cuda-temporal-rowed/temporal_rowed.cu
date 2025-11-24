#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"
#include "../_shared/cuda-helpers/block_to_2dim.hpp"
#include "../_shared/cuda-helpers/border_policies.cuh"
#include "./temporal_rowed.hpp"
#include <inttypes.h>

namespace algorithms {

using idx_t = std::int64_t;
template <typename word_type>
using GridModel = RowedGridOnCudaForTemporal<word_type>;

using border_policy_independent = border_policies::border_policy_independent;

namespace {

template <idx_t _block_y_size, idx_t _temporal_steps, idx_t _words_per_thread>
struct consts {
    static constexpr idx_t block_x_size = 32;
    static constexpr idx_t block_y_size = _block_y_size;
    static constexpr idx_t temporal_steps = _temporal_steps;
    static constexpr idx_t words_per_thread = _words_per_thread;

    static constexpr idx_t effective_x_block_size = block_x_size - 2;
    static constexpr idx_t effective_y_block_size = block_y_size * words_per_thread - 2 * (temporal_steps - 1);
    static constexpr idx_t x_mem_offset = 1;
    static constexpr idx_t y_mem_offset = temporal_steps;
    static constexpr idx_t shared_x_size = block_x_size;
    static constexpr idx_t shared_y_size = effective_y_block_size + 2 * temporal_steps;
    static constexpr idx_t shm_width = shared_x_size;
};

template <typename constants, typename border_policy, typename word_type>
__device__ __forceinline__ void load_to_shared(GridModel<word_type> data, word_type* shared) {
    using c = constants;
    
    idx_t x_start = blockIdx.x * c::effective_x_block_size - c::x_mem_offset;
    idx_t y_start = blockIdx.y * c::effective_y_block_size - c::y_mem_offset;

    idx_t shared_x = threadIdx.x;
    idx_t shared_y_start = threadIdx.y * c::words_per_thread;

    idx_t global_x = x_start + threadIdx.x;
    idx_t global_y_start = y_start + threadIdx.y * c::words_per_thread;

    for (idx_t i = 0; i < c::words_per_thread; ++i) {
        idx_t shared_idx = border_policy_independent::get_idx(shared_x, shared_y_start + i, c::shm_width);
        shared[shared_idx] = border_policy::load(global_x, global_y_start + i, data);
    }

    bool is_last_warp = (threadIdx.y == (blockDim.y - 1));

    if (is_last_warp) {
        for (idx_t i = 0; i < 2; ++i) {
            idx_t shared_idx = border_policy_independent::get_idx(shared_x, shared_y_start + c::words_per_thread + i, c::shm_width);
            shared[shared_idx] = border_policy::load(global_x, global_y_start + c::words_per_thread + i, data);
        }
    }
}

template <typename constants, typename word_type>
__device__ __forceinline__ void write_to_global(GridModel<word_type> data, word_type* shared) {
    using c = constants;

    idx_t constexpr x_lower_bound = c::x_mem_offset;
    idx_t constexpr x_upper_bound = x_lower_bound + c::effective_x_block_size;

    idx_t constexpr y_lower_bound = c::y_mem_offset;
    idx_t constexpr y_upper_bound = y_lower_bound + c::effective_y_block_size;

    idx_t x_start = blockIdx.x * c::effective_x_block_size - c::x_mem_offset;
    idx_t y_start = blockIdx.y * c::effective_y_block_size - c::y_mem_offset;

    idx_t shared_x = threadIdx.x;
    idx_t shared_y_start = threadIdx.y * c::words_per_thread + 1;

    idx_t global_x = x_start + threadIdx.x;
    idx_t global_y_start = y_start + threadIdx.y * c::words_per_thread + 1;

    for (idx_t i = 0; i < c::words_per_thread; ++i) {
        idx_t shared_y = shared_y_start + i;

        if (shared_y >= y_lower_bound && shared_y < y_upper_bound && 
            shared_x >= x_lower_bound && shared_x < x_upper_bound) {
            
            idx_t global_y = global_y_start + i;

            idx_t global_idx = border_policy_independent::get_idx(global_x, global_y, data.x_size);
            data.output[global_idx] = shared[border_policy_independent::get_idx(shared_x, shared_y, c::shm_width)];
        }
    }
}

template <typename bit_grid_mode, typename constants, typename word_type>
__device__ __forceinline__ void compute_GoL_block(word_type* tile) {
    using c = constants;

    auto shm_idx = [](idx_t x, idx_t y) {
        return y * c::shm_width + x;
    };

    const idx_t y_to_shm_start = c::words_per_thread * threadIdx.y + 1;
    const idx_t x_to_shm = threadIdx.x; // for 9 nei algs (+ 1)

    for (idx_t time_step = 0; time_step < c::temporal_steps; ++time_step) {
        
        word_type bottom_halo_word = tile[shm_idx(x_to_shm, y_to_shm_start + c::words_per_thread)];

        word_type ct = 0;
        word_type cc = tile[shm_idx(x_to_shm, y_to_shm_start - 1)];
        word_type cb = tile[shm_idx(x_to_shm, y_to_shm_start)];

        __syncthreads();

        for (idx_t y_offset = 0; y_offset < (c::words_per_thread - 1); ++y_offset) {
            const idx_t y_to_shm = y_to_shm_start + y_offset;

            ct = cc;
            cc = cb;
            cb = tile[shm_idx(x_to_shm, y_to_shm + 1)];

            word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(ct, cc, cb);
            tile[shm_idx(x_to_shm, y_to_shm)] = new_value;
        }

        ct = cc;
        cc = cb;
        cb = bottom_halo_word;

        word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(ct, cc, cb);
        tile[shm_idx(x_to_shm, y_to_shm_start + c::words_per_thread - 1)] = new_value;
        
        __syncthreads();
    }
}

template <typename bit_grid_mode, typename constants, typename word_type>
__device__ __forceinline__ void compute_GoL_block_with_border_elimination(word_type* tile) {
    using c = constants;

    auto shm_idx = [](idx_t x, idx_t y) {
        return y * c::shm_width + x;
    };

    const idx_t y_to_shm_initial_start = c::words_per_thread * threadIdx.y + 1;
    const idx_t x_to_shm = threadIdx.x; // for 9 nei algs (+ 1)

    for (idx_t time_step = 0; time_step < c::temporal_steps; ++time_step) {
        
        const idx_t y_to_shm_start = y_to_shm_initial_start + time_step;

        word_type ct = 0;
        word_type cc = tile[shm_idx(x_to_shm, y_to_shm_start - 1)];
        word_type cb = tile[shm_idx(x_to_shm, y_to_shm_start)];


        const idx_t inactive_warp_count = (2 * time_step) / c::words_per_thread;
        const idx_t active_warp_count = c::block_y_size - inactive_warp_count;
        const idx_t last_active_warp = active_warp_count - 1;

        if (threadIdx.y < last_active_warp) {

            word_type bottom_halo_word = tile[shm_idx(x_to_shm, y_to_shm_start + c::words_per_thread)];
            __syncthreads();
            
            for (idx_t y_offset = 0; y_offset < (c::words_per_thread - 1); ++y_offset) {
                const idx_t y_to_shm = y_to_shm_start + y_offset;

                ct = cc;
                cc = cb;
                cb = tile[shm_idx(x_to_shm, y_to_shm + 1)];

                word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(ct, cc, cb);
                tile[shm_idx(x_to_shm, y_to_shm)] = new_value;
            }

            ct = cc;
            cc = cb;
            cb = bottom_halo_word;

            word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(ct, cc, cb);
            tile[shm_idx(x_to_shm, y_to_shm_start + c::words_per_thread - 1)] = new_value;

        } else if (threadIdx.y == last_active_warp) {
            
            __syncthreads();

            const idx_t omitted_items = (2 * time_step) % c::words_per_thread;
            const idx_t items_computed = c::words_per_thread - omitted_items;

            for (idx_t y_offset = 0; y_offset < items_computed; ++y_offset) {
                const idx_t y_to_shm = y_to_shm_start + y_offset;

                ct = cc;
                cc = cb;
                cb = tile[shm_idx(x_to_shm, y_to_shm + 1)];

                word_type new_value = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(ct, cc, cb);
                tile[shm_idx(x_to_shm, y_to_shm)] = new_value;
            }
        }

        __syncthreads();
    }
}

template <typename bit_grid_mode, typename constants, typename border_policy, typename word_type>
__global__ __launch_bounds__(constants::block_x_size * constants::block_y_size) 
void game_of_live_kernel(GridModel<word_type> data) {

    __shared__ word_type shm_tile_buffer[constants::shared_x_size * constants::shared_y_size];

    load_to_shared<constants, border_policy>(data, shm_tile_buffer);
    __syncthreads();

    // compute_GoL_block<bit_grid_mode, constants>(shm_tile_buffer);
    compute_GoL_block_with_border_elimination<bit_grid_mode, constants>(shm_tile_buffer);
    __syncthreads();
    
    write_to_global<constants>(data, shm_tile_buffer);
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
template <idx_t block_y_size, idx_t temporal_steps, idx_t words_per_thread, typename border_policy>
void GoLCudaTemporalRowed<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) {
    using c = consts<block_y_size, temporal_steps, words_per_thread>;
    
    dim3 block = {c::block_x_size, c::block_y_size};
    
    if (cuda_data.x_size % c::effective_x_block_size != 0 || cuda_data.y_size % c::effective_y_block_size != 0) {
        throw std::runtime_error("Grid size must be divisible by block size - effective y block size is "
                                 + std::to_string(c::effective_y_block_size) + " and x block size is "
                                 + std::to_string(c::effective_x_block_size));
    }

    if ((c::temporal_steps - 1) % words_per_thread != 0) {
        throw std::runtime_error("Temporal steps minus 1 must be divisible by words per thread");
    }

    if (iterations % c::temporal_steps != 0) {
        throw std::runtime_error("Iterations must be divisible by temporal steps");
    }

    if (c::effective_y_block_size % words_per_thread != 0) {
        throw std::runtime_error("Effective y block size (" + std::to_string(c::effective_y_block_size) + ") must be divisible by words per thread (" + std::to_string(words_per_thread) + ")");
    }

    dim3 grid(cuda_data.x_size / c::effective_x_block_size, cuda_data.y_size / c::effective_y_block_size);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    auto big_steps = iterations / c::temporal_steps;

    for (std::size_t i = 0; i < big_steps; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i * c::temporal_steps;
            return;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }
        
        game_of_live_kernel<bit_grid_mode, c, border_policy><<<grid, block>>>(cuda_data);

        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::GoLCudaTemporalRowed<common::CHAR, 64, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaTemporalRowed<common::INT, 64, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaTemporalRowed<common::CHAR, 32, algorithms::WarpExchangeFullAdderOnRowsMode>;
template class algorithms::GoLCudaTemporalRowed<common::INT, 32, algorithms::WarpExchangeFullAdderOnRowsMode>;
