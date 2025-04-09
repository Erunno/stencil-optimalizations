#ifndef ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
#define ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
    
#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../cuda-helpers/shift.cuh"

namespace algorithms {



template <typename word_type>
struct WarpExchangeFullAdderOnRows {

    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    static __device__ __forceinline__  word_type compute_center_word(
        word_type ct, 
        word_type cc,
        word_type cb) {

        const word_type _0_center_only_neighbors = ct ^ cb;
        const word_type _1_center_only_neighbors = ct & cb;

        const word_type _0_center_full_column = _0_center_only_neighbors ^ cc;
        const word_type _1_center_full_column = _1_center_only_neighbors | (_0_center_only_neighbors & cc);

        const word_type _0_right = shift_val_within_warp<ShiftDirection::LEFT>(_0_center_full_column);
        const word_type _1_right = shift_val_within_warp<ShiftDirection::LEFT>(_1_center_full_column);

        const word_type _0_left = shift_val_within_warp<ShiftDirection::RIGHT>(_0_center_full_column);
        const word_type _1_left = shift_val_within_warp<ShiftDirection::RIGHT>(_1_center_full_column);

        const word_type _0_shifted_left = (_0_center_full_column << 1) | (_0_left >> (BITS - 1)); // 3 ops
        const word_type _1_shifted_left = (_1_center_full_column << 1) | (_1_left >> (BITS - 1)); // 3 ops

        const word_type _0_shifted_right = (_0_center_full_column >> 1) | (_0_right << (BITS - 1)); // 3 ops
        const word_type _1_shifted_right = (_1_center_full_column >> 1) | (_1_right << (BITS - 1)); // 3 ops

        // partial = 17 ops

        return from_7_bits_to_result(
            _0_shifted_left, _1_shifted_left,
            _0_center_only_neighbors, _1_center_only_neighbors,
            _0_shifted_right, _1_shifted_right,            
            cc); // 16 ops

        // total = 33 ops
    }

    static __device__ __forceinline__ word_type from_7_bits_to_result(
        word_type i1, word_type i2, word_type i3, word_type i4,
        word_type i5, word_type i6, word_type i7) {
        
        // Hidden layer 1
        const word_type h1_0 = i2 | i6;
        const word_type h1_1 = i7;
        const word_type h1_2 = i1 & i4;
        const word_type h1_3 = i3 | i4;
        const word_type h1_4 = i1;
        const word_type h1_5 = i2 & i6;
        const word_type h1_6 = i3 ^ i5;
        const word_type h1_7 = i1 ^ i3;
    
        // Hidden layer 2
        const word_type h2_0 = h1_1;
        const word_type h2_1 = h1_4 ^ h1_6;
        const word_type h2_2 = h1_6 & h1_7;
        const word_type h2_3 = h1_5;
        const word_type h2_4 = h1_0 ^ h1_3;
        const word_type h2_5 = h1_0 & h1_2;
    
        // Hidden layer 3
        const word_type h3_0 = h2_3 | h2_5;
        const word_type h3_1 = h2_0 | h2_1;
        const word_type h3_2 = h2_2 ^ h2_4;
    
        // Hidden layer 4
        const word_type h4_0 = h3_0 ^ h3_2;
        const word_type h4_1 = h3_1 & h3_2;
    
        // Output layer
        const word_type o1 = h4_0 & h4_1;

        return o1;
    }
};

}

#endif