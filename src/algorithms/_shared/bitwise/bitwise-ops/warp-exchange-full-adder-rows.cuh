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
        
        // .model spec
        // .inputs 1 2 3 4 5 6 7
        // .outputs 8
        // .names 1 3 5 22
        // 010 1
        // 101 1
        // .names 3 4 22 11
        // 001 1
        // 010 1
        // 100 1
        // 110 1
        // 111 1
        // .names 2 6 11 12
        // 001 1
        // 010 1
        // 100 1
        // .names 1 4 7 21
        // 001 1
        // 011 1
        // 100 1
        // 110 1
        // 111 1
        // .names 3 5 21 9
        // 010 1
        // 011 1
        // 100 1
        // 101 1
        // .names 1 21 9 10
        // 001 1
        // 010 1
        // 011 1
        // 100 1
        // 101 1
        // 110 1
        // .names 10 12 23
        // 11 1
        // .names 23 8
        // 1 1
        // .end

        
        
        if constexpr (BITS == 32) {
            constexpr unsigned int ta = 0xF0;
            constexpr unsigned int tb = 0xCC;
            constexpr unsigned int tc = 0xAA;

            // immLut has to be given as an immediate value

            // gate_22
            constexpr unsigned int immLut_22 = (~ta & tb & ~tc) | (ta & ~tb & tc);
            word_type gate_22;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0x24;"
                : "=r"(gate_22)
                : "r"(i1), "r"(i3), "r"(i5)
            );
            static_assert(immLut_22 == 0x24u, "LUT mismatch");

            // gate_11
            constexpr unsigned int immLut_11 = (~ta & ~tb & tc) | (~ta & tb & ~tc) | (ta & ~tb & ~tc) | (ta & tb & ~tc) | (ta & tb & tc);
            word_type gate_11;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0xd6;"
                : "=r"(gate_11)
                : "r"(i3), "r"(i4), "r"(gate_22)
            );
            static_assert(immLut_11 == 0xd6u, "LUT mismatch");

            // gate_12
            constexpr unsigned int immLut_12 = (~ta & ~tb & tc) | (~ta & tb & ~tc) | (ta & ~tb & ~tc);
            word_type gate_12;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0x16;"
                : "=r"(gate_12)
                : "r"(i2), "r"(i6), "r"(gate_11)
            );
            static_assert(immLut_12 == 0x16u, "LUT mismatch");

            // gate_21
            constexpr unsigned int immLut_21 = (~ta & ~tb & tc) | (~ta & tb & tc) | (ta & ~tb & ~tc) | (ta & tb & ~tc) | (ta & tb & tc);
            word_type gate_21;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0xda;"
                : "=r"(gate_21)
                : "r"(i1), "r"(i4), "r"(i7)
            );
            static_assert(immLut_21 == 0xdau, "LUT mismatch");

            // gate_9
            constexpr unsigned int immLut_9 = (~ta & tb & ~tc) | (~ta & tb & tc) | (ta & ~tb & ~tc) | (ta & ~tb & tc);
            word_type gate_9;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0x3c;"
                : "=r"(gate_9)
                : "r"(i3), "r"(i5), "r"(gate_21)
            );
            static_assert(immLut_9 == 0x3cu, "LUT mismatch");

            // gate_10
            constexpr unsigned int immLut_10 = (~ta & ~tb & tc) | (~ta & tb & ~tc) | (~ta & tb & tc) | (ta & ~tb & ~tc) | (ta & ~tb & tc) | (ta & tb & ~tc);
            word_type gate_10;
            asm (
                "lop3.b32 %0, %1, %2, %3, 0x7e;"
                : "=r"(gate_10)
                : "r"(i1), "r"(gate_21), "r"(gate_9)
            );
            static_assert(immLut_10 == 0x7eu, "LUT mismatch");

            // gate_23
            word_type gate_23 = gate_10 & gate_12;

            // output
            return gate_23;

        } else {
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
    }
};

}

#endif
