#ifndef SHIFT_CUH
#define SHIFT_CUH

#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>

namespace algorithms {

enum class ShiftDirection {
    LEFT,
    RIGHT
};

// template <ShiftDirection shift_direction, typename word_type>
// __device__ __forceinline__ word_type shift_val_within_warp(word_type value) {
//     constexpr int warpSize = 32;
//     auto lane_id = threadIdx.x % warpSize;

//     auto neighbor_id = (shift_direction == ShiftDirection::LEFT)
//         ? (lane_id + 1) % warpSize
//         : (lane_id - 1 + warpSize) % warpSize;
    
//     return __shfl_sync(0xFFFFFFFF, value, neighbor_id, warpSize);
// }
 
template <ShiftDirection shift_direction, typename word_type>
__device__ __forceinline__ word_type shift_val_within_warp(word_type value) {
    constexpr int FULL_MASK = ~0;
    constexpr int delta = 1;
    
    if constexpr (shift_direction == ShiftDirection::LEFT) {
        return __shfl_down_sync(FULL_MASK, value, delta);
    } else {
        return __shfl_up_sync(FULL_MASK, value, delta);
    }
}

}

#endif