#ifndef CUDA_NAIVE_LOCAL_CU
#define CUDA_NAIVE_LOCAL_CU

#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "../../infrastructure/timer.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_local.hpp"
#include "gol_cuda_naive_just_tiling.hpp"
#include <cuda_runtime.h>
#include "../_shared/common_grid_types.hpp"
#include "../_shared/common_grid_types.hpp"
#include "./tiling-policies.cuh"
#include "./gol-tile-computation.cuh"

namespace algorithms::cuda_naive_local {

namespace {

template <typename tiling_policy, typename word_type, typename state_store_type>
__device__ __forceinline__ void cpy_to_output(
    WarpInfo info, BitGridWithChangeInfo<word_type, state_store_type> data) {

    auto x_fst = info.x_abs_start;
    auto x_lst = info.x_abs_start + tiling_policy::X_COMPUTED_WORD_COUNT;
    auto y_fst = info.y_abs_start;
    auto y_lst = info.y_abs_start + tiling_policy::Y_COMPUTED_WORD_COUNT;

    for (idx_t x = x_fst; x < x_lst; ++x) {
        for (idx_t y = y_fst; y < y_lst; ++y) {
            data.output[get_idx(x, y, data.x_size)] = data.input[get_idx(x, y, data.x_size)];
        }
    }
}


template <typename state_store_type>
__device__ __forceinline__ void set_changed_state_for_block(
    WarpInfo info, shm_private_value_t* block_store, state_store_type* global_store) {
        
    auto tiles = blockDim.x / WARP_SIZE;
    state_store_type result = 0;

    for (int i = 0; i < tiles; ++i) {
        state_store_type val = block_store[i] ? 1 : 0;
        result |= val << i;
    }

    global_store[blockIdx.x] = result;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ bool warp_tile_changed(
    idx_t x_tile, idx_t y_tile, 
    WarpInfo info, state_store_type* cached_store) {

    x_tile += tiling_policy::BLOCK_TILE_DIM_X;
    y_tile += tiling_policy::BLOCK_TILE_DIM_Y;

    auto x_word_idx = x_tile / tiling_policy::BLOCK_TILE_DIM_X;
    auto y_word_idx = y_tile / tiling_policy::BLOCK_TILE_DIM_Y;

    auto word = cached_store[y_word_idx * StateStoreInfo<state_store_type>::CACHE_SIZE_X + x_word_idx];

    auto x_bit_idx = x_tile % tiling_policy::BLOCK_TILE_DIM_X;
    auto y_bit_idx = y_tile % tiling_policy::BLOCK_TILE_DIM_Y;

    state_store_type one = 1;
    auto mask = one << (y_bit_idx * tiling_policy::BLOCK_TILE_DIM_X + x_bit_idx);

    return word & mask;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ bool tile_or_neighbours_changed(
    idx_t x_tile, idx_t y_tile,
    WarpInfo info, state_store_type* cached_store) {

    for(idx_t y = y_tile - 1; y <= y_tile + 1; ++y) {
        for(idx_t x = x_tile - 1; x <= x_tile + 1; ++x) {
            if (warp_tile_changed<tiling_policy>(x, y, info, cached_store)) {
                return true;
            }
        }
    }

    return false;
}

template <typename tiling_policy, typename state_store_type>
__device__ __forceinline__ void load_state_store(
    WarpInfo info,
    state_store_type* store, state_store_type* shared_store) {
    
    auto idx = threadIdx.x % StateStoreInfo<state_store_type>::CACHE_SIZE;
    
    auto x_block_to_load = idx % StateStoreInfo<state_store_type>::CACHE_SIZE_X + info.x_block - 1;
    auto y_block_to_load = idx / StateStoreInfo<state_store_type>::CACHE_SIZE_X + info.y_block - 1;

    if (x_block_to_load >= info.x_block_count || y_block_to_load >= info.y_block_count) {
        shared_store[idx] = 0;
        return;
    }

    auto store_idx = y_block_to_load * info.x_block_count + x_block_to_load;

    shared_store[idx] = store[store_idx];
}

template <typename tiling_policy, typename word_type, typename state_store_type>
__device__ __forceinline__ void prefetch_state_stores(
    WarpInfo info,
    BitGridWithChangeInfo<word_type, state_store_type> data,
    state_store_type* cache_last, state_store_type* cache_before_last) {
     
    if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE) {
        load_state_store<tiling_policy>(info, data.change_state_store.before_last, cache_before_last);
    }
    else if (threadIdx.x < StateStoreInfo<state_store_type>::CACHE_SIZE * 2) {
        load_state_store<tiling_policy>(info, data.change_state_store.last, cache_last);
    }
}

template <StreamingDir DIRECTION, typename bit_grid_mode, typename tiling_policy, typename word_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<word_type, state_store_type> data) {

    extern __shared__ shm_private_value_t block_store[];
    __shared__ state_store_type change_state_last[StateStoreInfo<state_store_type>::CACHE_SIZE];
    __shared__ state_store_type change_state_before_last[StateStoreInfo<state_store_type>::CACHE_SIZE];
    
    WarpInfo info = tiling_policy::template get_warp_info<idx_t>(data.x_size, data.y_size);
    prefetch_state_stores<tiling_policy>(info, data, change_state_last, change_state_before_last);

    __syncthreads();

    bool entire_tile_changed;

    if (!tile_or_neighbours_changed<tiling_policy>(info.x_warp, info.y_warp, info, change_state_last)) {
        
        if (warp_tile_changed<tiling_policy>(info.x_warp, info.y_warp, info, change_state_before_last)) {
            cpy_to_output<tiling_policy>(info, data);
        }
        
        entire_tile_changed = false;
    }
    else {
        auto local_tile_changed = compute_GOL_on_tile<word_type, bit_grid_mode, DIRECTION, tiling_policy>(info, data);
        entire_tile_changed = __any_sync(0xFF'FF'FF'FF, local_tile_changed);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        block_store[threadIdx.x / WARP_SIZE] = entire_tile_changed ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        set_changed_state_for_block(info, block_store, data.change_state_store.current);
    }
}

} // namespace

template <typename grid_cell_t, std::size_t Bits, typename state_store_type, typename bit_grid_mode>
template <StreamingDir DIRECTION, typename tiling_policy>
void GoLCudaNaiveLocalWithState<grid_cell_t, Bits, state_store_type, bit_grid_mode>::run_kernel(size_type iterations) {

    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

    auto warp_tile_per_block = block_size / WARP_SIZE;
    auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        cudaEventRecord(events[i], stream);

        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }

        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
            rotate_state_stores();      
        }

        game_of_live_kernel<DIRECTION, bit_grid_mode, tiling_policy><<<blocks, block_size, shm_size, stream>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }

    cudaEventRecord(events[_performed_iterations], stream);
}

// -----------------------------------------------------------------------------------------------
// JUST TILING KERNEL
// -----------------------------------------------------------------------------------------------

template <StreamingDir DIRECTION, typename bit_grid_mode, typename tiling_policy, typename word_type>
__global__ void game_of_live_kernel_just_tiling(BitGridWithTiling<word_type> data) {
    WarpInfo info = tiling_policy::template get_warp_info<idx_t>(data.x_size, data.y_size);
    compute_GOL_on_tile<word_type, bit_grid_mode, DIRECTION, tiling_policy>(info, data);
}


template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
template <StreamingDir DIRECTION, typename tiling_policy>
void GoLCudaNaiveJustTiling<grid_cell_t, Bits, bit_grid_mode>::run_kernel(size_type iterations) {
    auto block_size = thread_block_size;
    auto blocks = get_thread_block_count();

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

        game_of_live_kernel_just_tiling<DIRECTION, bit_grid_mode, tiling_policy><<<blocks, block_size>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::CHAR, 64, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocalWithState<common::INT, 64, std::uint64_t, algorithms::BitTileMode>;

template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitColumnsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitTileMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::BitWastefulRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::HalfPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::FullyPackedRowsMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 16, algorithms::FujitaMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 32, algorithms::FujitaMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::CHAR, 64, algorithms::FujitaMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 16, algorithms::FujitaMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 32, algorithms::FujitaMode>;
template class algorithms::cuda_naive_local::GoLCudaNaiveJustTiling<common::INT, 64, algorithms::FujitaMode>;

#endif // CUDA_NAIVE_LOCAL_CU
