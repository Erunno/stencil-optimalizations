#ifndef ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
#define ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
    
#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../cuda-helpers/shift.cuh"
#include <algorithm>
#include <ranges>
#include <type_traits>

namespace algorithms {

// wrapper for truth rows encoded as unsigned int that supports only bitwise OR operation
template<unsigned int data>
requires  (data <= 0xFF) && (data >= 0)
struct lut3 : std::integral_constant<unsigned int, data> {
    template<unsigned int rhs_data>
    __host__ __device__ constexpr auto operator|(lut3<rhs_data>) -> lut3<data | rhs_data> {
        return {};
    }
};

template<char... row>
constexpr auto operator""_tr() {
    constexpr unsigned int ta = 0xF0;
    constexpr unsigned int tb = 0xCC;
    constexpr unsigned int tc = 0xAA;

    constexpr char arr[] = {row...};
    static_assert(sizeof(arr) == 3, "Truth row must be exactly 3 characters long.");
    static_assert(std::all_of(std::begin(arr), std::end(arr), [](char c){ return c == '0' || c == '1'; }), "Truth row can only contain '0' and '1' characters.");

    constexpr unsigned int immLut = (arr[0] == '1' ? ta : ~ta) &
                                    (arr[1] == '1' ? tb : ~tb) &
                                    (arr[2] == '1' ? tc : ~tc) & 0xFF;

    return lut3<immLut>{};
}

template<typename word_type, unsigned int immLut>
requires (sizeof(word_type) == 4)
__device__ __forceinline__
word_type run_lop3_gate(
    const word_type in1,
    const word_type in2,
    const word_type in3,
    lut3<immLut>) {
    word_type out;
    asm (
        "lop3.b32 %0, %1, %2, %3, %4;"
        : "=r"(out)
        : "r"(in1), "r"(in2), "r"(in3), "n"(immLut)
    );
    return out;
}

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

        static_assert(BITS == 32 || BITS == 64, "Only 32-bit and 64-bit word types are supported.");

        // .model spec
        // .inputs 1 2 3 4 5 6 7
        // .outputs 12
        // .names 1 3 5 18
        // 001 1
        // 010 1
        // 100 1
        // 111 1
        // .names 1 5 18 8
        // 010 1
        // 100 1
        // 110 1
        // 111 1
        // .names 4 8 18 11
        // 010 1
        // 011 1
        // 100 1
        // 101 1
        // 110 1
        // 111 1
        // .names 2 6 11 9
        // 001 1
        // 010 1
        // 100 1
        // .names 1 4 7 10
        // 001 1
        // 011 1
        // 101 1
        // .names 9 18 10 12
        // 101 1
        // 110 1
        // 111 1
        // .end

        constexpr auto compute_32 = []<typename T>(T i1, T i2, T i3, T i4, T i5, T i6, T i7) -> T {
            // (.names == input1 input2 input3 -> output gate)
            // .names 1 3 5 18
            // (truth table; ignore the last column)
            // 001 1
            // 010 1
            // 100 1
            // 111 1
            T gate_18 = run_lop3_gate(i1, i3, i5, 001_tr | 010_tr | 100_tr | 111_tr);

            T gate_8 = run_lop3_gate(i1, i5, gate_18, 010_tr | 100_tr | 110_tr | 111_tr);
            T gate_11 = run_lop3_gate(i4, gate_8, gate_18, 010_tr | 011_tr | 100_tr | 101_tr | 110_tr | 111_tr);
            T gate_9 = run_lop3_gate(i2, i6, gate_11, 001_tr | 010_tr | 100_tr);
            T gate_10 = run_lop3_gate(i1, i4, i7, 001_tr | 011_tr | 101_tr);
            T gate_12 = run_lop3_gate(gate_9, gate_18, gate_10, 101_tr | 110_tr | 111_tr);
            return gate_12;
        };

        if constexpr (BITS == 32) {
            return compute_32(i1, i2, i3, i4, i5, i6, i7);
        } else if constexpr (BITS == 64) {
            return compute_32(
                static_cast<uint32_t>(i1),
                static_cast<uint32_t>(i2),
                static_cast<uint32_t>(i3),
                static_cast<uint32_t>(i4),
                static_cast<uint32_t>(i5),
                static_cast<uint32_t>(i6),
                static_cast<uint32_t>(i7)
            ) |
            (static_cast<uint64_t>(compute_32(
                static_cast<uint32_t>(i1 >> 32),
                static_cast<uint32_t>(i2 >> 32),
                static_cast<uint32_t>(i3 >> 32),
                static_cast<uint32_t>(i4 >> 32),
                static_cast<uint32_t>(i5 >> 32),
                static_cast<uint32_t>(i6 >> 32),
                static_cast<uint32_t>(i7 >> 32)
            )) << 32);
        }
    }
};

}

#endif
