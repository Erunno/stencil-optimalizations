#include "cuda_local_one_cell.hpp"
#include "../_shared/bitwise/bitwise-ops/cuda-ops-interface.cuh"
#include "../_shared/bitwise/bitwise-ops/macro-cols.hpp"
#include "../_shared/bitwise/bit_modes.hpp"
#include "../../infrastructure/timer.hpp"
#include "./models.hpp"
#include <cuda_runtime.h>
#include "../_shared/common_grid_types.hpp"

namespace algorithms::cuda_local_one_cell {

using shm_private_value_t = std::uint32_t;
using idx_t = std::int64_t;

constexpr std::size_t BLOCK_X_DIM = 32;

__device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename word_type, typename CudaData>
__device__ __forceinline__ word_type load(idx_t x, idx_t y, CudaData data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

__device__ __forceinline__ auto get_linear_block_idx() {
    return blockIdx.y * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ auto get_linear_thread_idx() {
    return threadIdx.y * blockDim.x + threadIdx.x;
}

struct warp_tile_info {
    idx_t x;
    idx_t y;
};

__device__ __forceinline__ warp_tile_info get_warp_tile_coords() {
    return {
        .x = 0,
        .y = threadIdx.y
    };
}

__device__ __forceinline__ auto get_thread_block_count() {
    return gridDim.x * gridDim.y;
}

template <typename word_type, typename bit_grid_mode, typename state_store_type>
__device__ __forceinline__ bool compute_GOL_on_tile(
    BitGridWithChangeInfo<word_type, state_store_type> data) {

    idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    idx_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= data.x_size || y >= data.y_size)
        return false;

    word_type lt = load<word_type>(x - 1, y - 1, data);
    word_type ct = load<word_type>(x + 0, y - 1, data);
    word_type rt = load<word_type>(x + 1, y - 1, data);
    word_type lc = load<word_type>(x - 1, y + 0, data);
    word_type cc = load<word_type>(x + 0, y + 0, data);
    word_type rc = load<word_type>(x + 1, y + 0, data);
    word_type lb = load<word_type>(x - 1, y + 1, data);
    word_type cb = load<word_type>(x + 0, y + 1, data);
    word_type rb = load<word_type>(x + 1, y + 1, data);

    auto new_cc = CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb); 
    auto tile_changed = new_cc != cc;

    data.output[get_idx(x, y, data.x_size)] = new_cc;

    return tile_changed;
}

template <typename word_type, typename state_store_type>
__device__ __forceinline__ void cpy_to_output(
    BitGridWithChangeInfo<word_type, state_store_type> data) {

    idx_t x = threadIdx.x + blockIdx.x * blockDim.x;
    idx_t y = threadIdx.y + blockIdx.y * blockDim.y;

    data.output[get_idx(x, y, data.x_size)] = data.input[get_idx(x, y, data.x_size)];
}

template <typename state_store_type>
__device__ __forceinline__ void set_changed_state_for_block(
    shm_private_value_t* block_store, state_store_type* global_store) {
    
    auto tiles = blockDim.y;
    auto thread_value = (threadIdx.x < tiles) ? block_store[threadIdx.x] : 0;

    auto result = __ballot_sync(0xFFFFFFFF, thread_value);
    
    if (threadIdx.x == 0) {
        global_store[get_linear_block_idx()] = result;
    }
}

template <typename state_store_type>
__device__ __forceinline__ bool warp_tile_changed(
    idx_t x_tile, idx_t y_tile,
    state_store_type* cached_store) {

    auto x_word_idx = x_tile + 1;
    auto y_word_idx = (y_tile + blockDim.y) / blockDim.y;

    auto word_idx = y_word_idx * StateStoreInfo::CACHE_SIZE_X + x_word_idx;
    auto bit = (y_tile + blockDim.y) % blockDim.y;

    return (cached_store[word_idx] >> bit) & 1;
}

template <typename state_store_type>
__device__ __forceinline__ bool tile_or_neighbours_changed(
    idx_t x_tile, idx_t y_tile,
    state_store_type* cached_store) {

    for(idx_t y = y_tile - 1; y <= y_tile + 1; ++y) {
        for(idx_t x = x_tile - 1; x <= x_tile + 1; ++x) {
            if (warp_tile_changed(x, y, cached_store)) {
                return true;
            }
        }
    }

    return false;
}

template <typename state_store_type, typename CudaData>
__device__ __forceinline__ void load_state_store(
    CudaData data,
    state_store_type* store, state_store_type* shared_store) {
    
    idx_t offset = get_linear_thread_idx() % StateStoreInfo::CACHE_SIZE;

    idx_t x_off = offset % StateStoreInfo::CACHE_SIZE_X - 1;
    idx_t y_off = offset / StateStoreInfo::CACHE_SIZE_X - 1;

    if (blockIdx.x + x_off < 0 || blockIdx.x + x_off >= gridDim.x ||
        blockIdx.y + y_off < 0 || blockIdx.y + y_off >= gridDim.y) {
        shared_store[offset] = 0;
        return;
    }

    idx_t store_idx =  get_linear_block_idx() + (y_off * gridDim.x + x_off);

    shared_store[offset] = store[store_idx];
}

template <typename word_type, typename state_store_type>
__device__ __forceinline__ void prefetch_state_stores(
    BitGridWithChangeInfo<word_type, state_store_type> data,
    state_store_type* cache_last, state_store_type* cache_before_last) {

    auto thread_idx = get_linear_thread_idx();

    if (thread_idx < StateStoreInfo::CACHE_SIZE) {
        load_state_store(data, data.change_state_store.last, cache_last);
    }
    else if (thread_idx < StateStoreInfo::CACHE_SIZE * 2) {
        load_state_store(data, data.change_state_store.current, cache_before_last);
    }
}

template <typename bit_grid_mode, typename word_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<word_type, state_store_type> data) {

    extern __shared__ shm_private_value_t block_store[];
    __shared__ state_store_type change_state_last[StateStoreInfo::CACHE_SIZE];
    __shared__ state_store_type change_state_before_last[StateStoreInfo::CACHE_SIZE];

    prefetch_state_stores(data, change_state_last, change_state_before_last);

    __syncthreads();

    bool entire_tile_changed;
    
    auto warp_tile = get_warp_tile_coords(); 

    if (!tile_or_neighbours_changed(warp_tile.x, warp_tile.y, change_state_last)) {
        
        if (warp_tile_changed(warp_tile.x, warp_tile.y, change_state_before_last)) {
            cpy_to_output(data);
        }
        
        entire_tile_changed = false;
    }
    else {
        auto local_tile_changed = compute_GOL_on_tile<word_type, bit_grid_mode>(data);
        entire_tile_changed = __any_sync(0xFF'FF'FF'FF, local_tile_changed);
    }

    if (threadIdx.x == 0) {
        block_store[threadIdx.y] = entire_tile_changed ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
        set_changed_state_for_block(block_store, data.change_state_store.current);
    }
}

template <typename grid_cell_t, std::size_t Bits, typename state_store_type, typename bit_grid_mode>
template <bool RECORD_CHANGED_TILES>
void GoLCudaLocalOneCellImpl<grid_cell_t, Bits, state_store_type, bit_grid_mode>::run_kernel(size_type iterations) {
    dim3 block = {static_cast<unsigned int>(BLOCK_X_DIM), static_cast<unsigned int>(this->params.thread_block_size / BLOCK_X_DIM)};
    dim3 grid(cuda_data.x_size / block.x, cuda_data.y_size / block.y);

    auto blocks = get_thread_block_count();

    auto warp_tile_per_block =  this->params.thread_block_size / WARP_SIZE;
    auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }

        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
            rotate_state_stores();      
        }

        if constexpr (RECORD_CHANGED_TILES) {
            record_changed_tiles(i);
        }

        game_of_live_kernel<bit_grid_mode><<<grid, block, shm_size>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms::cuda_local_one_cell

template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint16_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint32_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint64_t, algorithms::BitColumnsMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::CHAR, 64, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 16, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 32, std::uint64_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint16_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint32_t, algorithms::BitTileMode>;
template class algorithms::cuda_local_one_cell::GoLCudaLocalOneCellImpl<common::INT, 64, std::uint64_t, algorithms::BitTileMode>;
