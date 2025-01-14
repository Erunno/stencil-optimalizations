#ifndef CUDA_NAIVE_LOCAL_CU
#define CUDA_NAIVE_LOCAL_CU

#include "../_shared/bitwise-cols/bitwise_ops_cuda_bit_ops.cuh"
#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include "./models.hpp"
#include "gol_cuda_naive_local.hpp"
#include <cuda_runtime.h>

namespace algorithms::cuda_naive_local {

using idx_t = std::int64_t;
// using idx_t = std::int32_t;
using shm_private_value_t = std::uint32_t;

constexpr std::size_t WARP_SIZE = 32;

template <typename col_type, typename state_store_type>
using WarpInfo = algorithms::cuda_naive_local::WarpInformation<col_type, state_store_type, idx_t>;

__device__ static __forceinline__ idx_t get_idx(idx_t x, idx_t y, idx_t x_size) {
    return y * x_size + x;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ WarpInfo<col_type, state_store_type> get_warp_info(
    const BitGridWithChangeInfo<col_type, state_store_type>& data) {
    WarpInfo<col_type, state_store_type> info;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    info.warp_idx = idx / WARP_SIZE;
    info.lane_idx = idx % WARP_SIZE;

    info.x_tiles = data.x_size / data.warp_tile_dims.x;
    info.y_tiles = data.y_size / data.warp_tile_dims.y;

    info.x_tile = info.warp_idx % info.x_tiles;
    info.y_tile = info.warp_idx / info.x_tiles;

    info.x_in_warp = info.lane_idx % data.warp_dims.x;
    info.y_in_warp = info.lane_idx / data.warp_dims.x;

    info.x_cols_in_warp = data.warp_tile_dims.x / data.warp_dims.x;
    info.y_rows_in_warp = data.warp_tile_dims.y / data.warp_dims.y;

    info.x_start = (info.x_tile * data.warp_tile_dims.x) + (info.x_in_warp * info.x_cols_in_warp);
    info.y_start = (info.y_tile * data.warp_tile_dims.y) + (info.y_in_warp * info.y_rows_in_warp);

    return info;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ col_type load(idx_t x, idx_t y,
                                                BitGridWithChangeInfo<col_type, state_store_type> data) {
    if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
        return 0;

    return data.input[get_idx(x, y, data.x_size)];
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ idx_t store_idx(idx_t x_tile, idx_t y_tile,
                                                  const WarpInfo<col_type, state_store_type>& info) {
    auto tile_idx = y_tile * info.x_tiles + x_tile;
    auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

    auto store_idx = tile_idx / warp_tiles_in_block;
    return store_idx;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ int store_bit_idx(idx_t x_tile, idx_t y_tile,
                                                    const WarpInfo<col_type, state_store_type>& info) {
    auto tile_idx = y_tile * info.x_tiles + x_tile;
    auto warp_tiles_in_block = blockDim.x / WARP_SIZE;

    return tile_idx % warp_tiles_in_block;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ bool warp_tile_changed(idx_t x_tile, idx_t y_tile,
                                                         const WarpInfo<col_type, state_store_type>& info,
                                                         state_store_type* store) {

    auto word = store[store_idx(x_tile, y_tile, info)];
    auto mask = 1 << store_bit_idx(x_tile, y_tile, info);

    return word & mask;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ bool tile_or_neighbors_changed(idx_t x_tile, idx_t y_tile,
                                                                 const WarpInfo<col_type, state_store_type>& info,
                                                                 state_store_type* store) {

    idx_t x_start = max(static_cast<idx_t>(0), x_tile - 1);
    idx_t y_start = max(static_cast<idx_t>(0), y_tile - 1);
    idx_t x_end = min(info.x_tiles, x_tile + 1);
    idx_t y_end = min(info.y_tiles, y_tile + 1);

    for (idx_t y = y_start; y <= y_end + 1; ++y) {
        for (idx_t x = x_start; x <= x_end; ++x) {
            if (warp_tile_changed(x, y, info, store)) {
                return true;
            }
        }
    }

    return false;
}

template <typename state_store_type>
__device__ static __forceinline__ void set_changed_state_for_block(shm_private_value_t* block_store,
                                                                   state_store_type* global_store) {

    auto tiles = blockDim.x / WARP_SIZE;
    state_store_type result = 0;

    for (int i = 0; i < tiles; ++i) {
        state_store_type val = block_store[i] ? 1 : 0;
        result |= val << i;
    }

    global_store[blockIdx.x] = result;
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ void cpy_to_output(const WarpInfo<col_type, state_store_type>& info,
                                                     const BitGridWithChangeInfo<col_type, state_store_type>& data) {

    for (idx_t y = info.y_start; y < info.y_start + info.y_rows_in_warp; ++y) {
        for (idx_t x = info.x_start; x < info.x_start + info.x_cols_in_warp; ++x) {
            data.output[get_idx(x, y, data.x_size)] = data.input[get_idx(x, y, data.x_size)];
        }
    }
}

template <typename col_type, typename state_store_type>
__device__ static __forceinline__ bool compute_GOL_on_tile(
    const WarpInfo<col_type, state_store_type>& info, const BitGridWithChangeInfo<col_type, state_store_type>& data) {

    bool tile_changed = false;

    for (idx_t y = info.y_start; y < info.y_start + info.y_rows_in_warp; ++y) {
        idx_t x = info.x_start;

        col_type lt, ct, rt;
        col_type lc, cc, rc;
        col_type lb, cb, rb;

        ct = load(x - 1, y - 1, data);
        cc = load(x - 1, y + 0, data);
        cb = load(x - 1, y + 1, data);

        rt = load(x + 0, y - 1, data);
        rc = load(x + 0, y + 0, data);
        rb = load(x + 0, y + 1, data);

        for (; x < info.x_start + info.x_cols_in_warp; ++x) {

            lt = ct;
            lc = cc;
            lb = cb;

            ct = rt;
            cc = rc;
            cb = rb;

            rt = load(x + 1, y - 1, data);
            rc = load(x + 1, y + 0, data);
            rb = load(x + 1, y + 1, data);

            auto new_cc = CudaBitwiseOps<col_type>::compute_center_col(lt, ct, rt, lc, cc, rc, lb, cb, rb);

            if (new_cc != cc) {
                tile_changed = true;
            }

            data.output[get_idx(x, y, data.x_size)] = new_cc;
        }
    }

    return tile_changed;
}

template <typename col_type, typename state_store_type>
__global__ void game_of_live_kernel(BitGridWithChangeInfo<col_type, state_store_type> data) {

    extern __shared__ shm_private_value_t block_store[];
    bool entire_tile_changed;

    auto info = get_warp_info(data);

    if (!tile_or_neighbors_changed(info.x_tile, info.y_tile, info, data.change_state_store.last)) {

        if (warp_tile_changed(info.x_tile, info.y_tile, info, data.change_state_store.before_last)) {
            cpy_to_output(info, data);
        }

        entire_tile_changed = false;
    }
    else {
        auto local_tile_changed = compute_GOL_on_tile(info, data);
        entire_tile_changed = __any_sync(0xFF'FF'FF'FF, local_tile_changed);
    }

    if (info.lane_idx == 0) {
        auto tiles_per_block = blockDim.x / WARP_SIZE;
        block_store[info.warp_idx % tiles_per_block] = entire_tile_changed ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        set_changed_state_for_block(block_store, data.change_state_store.current);
    }
}

template <std::size_t Bits, typename state_store_type>
void GoLCudaNaiveLocal<Bits, state_store_type>::run_kernel(size_type iterations) {
    const auto block_size = thread_block_size;
    const auto blocks = get_thread_block_count();

    const auto warp_tile_per_block = block_size / WARP_SIZE;
    const auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

    for (std::size_t i = 0; i < iterations; ++i) {
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
            rotate_state_stores();
        }

        game_of_live_kernel<<<blocks, block_size, shm_size>>>(cuda_data);
        // check_state_stores();
    }
}

// template <std::size_t Bits, typename state_store_type>
// void GoLCudaNaiveLocal<Bits, state_store_type>::run_kernel(size_type iterations) {
//     auto block_size = thread_block_size;
//     auto blocks = get_thread_block_count();
//     auto warp_tile_per_block = block_size / WARP_SIZE;
//     auto shm_size = warp_tile_per_block * sizeof(shm_private_value_t);

//     cudaGraph_t graph;
//     cudaGraphExec_t instance;
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

//     for (std::size_t i = 0; i < iterations; ++i) {
//         if (i != 0) {
//             std::swap(cuda_data.input, cuda_data.output);
//             rotate_state_stores();
//         }
//         game_of_live_kernel<<<blocks, block_size, shm_size, stream>>>(cuda_data);
//     }

//     cudaStreamEndCapture(stream, &graph);
//     cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

//     cudaGraphLaunch(instance, stream);
//     cudaStreamSynchronize(stream);

//     cudaStreamDestroy(stream);
// }

} // namespace algorithms::cuda_naive_local

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<16, std::uint64_t>;

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<32, std::uint64_t>;

template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint16_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint32_t>;
template class algorithms::cuda_naive_local::GoLCudaNaiveLocal<64, std::uint64_t>;

#endif // CUDA_NAIVE_LOCAL_CU
