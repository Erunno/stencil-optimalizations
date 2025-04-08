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

        word_type _0_center, _1_center;
        add_three_bits(
            ct, cc, cb,
            _0_center, _1_center); // 5 ops

        word_type _0_right = shift_val_within_warp<ShiftDirection::LEFT>(_0_center);
        word_type _1_right = shift_val_within_warp<ShiftDirection::LEFT>(_1_center);

        word_type _0_left = shift_val_within_warp<ShiftDirection::RIGHT>(_0_center);
        word_type _1_left = shift_val_within_warp<ShiftDirection::RIGHT>(_1_center);

        word_type _0_shifted_left = (_0_center << 1) | (_0_left >> (BITS - 1)); // 3 ops
        word_type _1_shifted_left = (_1_center << 1) | (_1_left >> (BITS - 1)); // 3 ops

        word_type _0_shifted_right = (_0_center >> 1) | (_0_right << (BITS - 1)); // 3 ops
        word_type _1_shifted_right = (_1_center >> 1) | (_1_right << (BITS - 1)); // 3 ops

        // partial = 17 ops

        word_type r_0 = ct ^ cb, r_1 = ct & cb, r_2 = 0; // 2 ops -- ! can be for free (computed in add_three_bits)

        add_two(
            _0_shifted_left, _1_shifted_left,
            r_0, r_1, r_2); // 8 ops
        add_two(
            _0_shifted_right, _1_shifted_right,
            r_0, r_1, r_2); // 8 ops

        // word_type r_0 = 0, r_1 = 0, r_2 = 0;
        // add_three_two_bits(
        //     _0_shifted_left, _1_shifted_left,
        //     _0_shifted_right, _1_shifted_right,
        //     _0_center, _1_center,
        //     r_0, r_1, r_2); // 13 ops
            
        // return GOL(cc, r_0, r_1, r_2); // 9 ops
        return GOL_8(cc, r_0, r_1, r_2); // 4 ops

        // total 37 ops
    }

    static __device__ __forceinline__ void add_one(word_type a, word_type& _0, word_type& _1) {
        word_type r_0 = a ^ _0;

        word_type _0_carry = a & _0;
        word_type r_1 = _0_carry ^ _1;

        _0 = r_0;
        _1 = r_1;
    }

    static __device__ __forceinline__ void add_three_bits(
        word_type a, word_type b, word_type c,
        word_type& _0, word_type& _1) {

        word_type axb = a ^ b;

        _0 = axb ^ c;
        _1 = (a & b) | (axb & c);
    }

    static __device__ __forceinline__ void add_two(
        word_type _0a, word_type _1a,
        word_type& _0, word_type& _1, word_type& _2
    ){
        word_type _1_xor_1a = _1 ^ _1a;

        word_type r_0 = _0a ^ _0;

        word_type _0_carry = _0a & _0;
        word_type r_1 = _1_xor_1a ^ _0_carry;

        word_type _1_carry = (_1 & _1a) | (_0_carry & (_1_xor_1a));

        _0 = r_0;
        _1 = r_1;
        _2 = _2 ^ _1_carry;
    }

    static __device__ __forceinline__ void add_three_two_bits(
    word_type a0, word_type a1, 
    word_type b0, word_type b1, 
    word_type c0, word_type c1,
    word_type& s2, word_type& s1, word_type& s0) {
    
    // Compute LSB sum and carry
    word_type a0xb0 = a0 ^ b0;
    s0 = a0xb0 ^ c0;
    word_type carry = (a0 & b0) | (a0xb0 & c0);

    // Compute intermediate values for MSB calculations
    word_type a1xb1 = a1 ^ b1;
    word_type sum_msb = a1xb1 ^ c1;
    word_type carry_msb = (a1 & b1) | (a1xb1 & c1);

    // Compute final s1 and s2
    s1 = sum_msb ^ carry;
    s2 = carry_msb | (carry & sum_msb);
}

    constexpr static word_type ones = ~static_cast<word_type>(0);
    constexpr static word_type zeros = 0;

    static __device__ __forceinline__ word_type GOL(
        word_type alive, word_type _0, word_type _1, word_type _2) {

        auto is_3 = _0 & _1 & (_2 ^ ones);
        auto is_4 = (_0 ^ ones) & (_1 ^ ones) & _2;

        return is_3 | (is_4 & alive);

        // total 9 ops
    }

    static __device__ __forceinline__ word_type GOL_8(
        word_type alive, word_type _0, word_type _1, word_type _2) {

        return (_0 | alive) & _1 & (_2 ^ ones);

        // total 4 ops
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