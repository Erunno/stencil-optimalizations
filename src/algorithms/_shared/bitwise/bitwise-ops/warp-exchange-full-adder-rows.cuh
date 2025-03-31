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

    
    constexpr static int BITS = sizeof(word_type) * 8;

    static __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        // auto lane_id = threadIdx.x % 32;
        // return  shift_val_within_warp<ShiftDirection::RIGHT>(lane_id == 3) * ~static_cast<word_type>(0);

        word_type _0 = cc;
        word_type _1 = 0; 
        word_type _2 = 0;

        add_one_bit(ct, _0, _1); // 3 ops
        add_one_bit(cb, _0, _1); // 3 ops

        word_type _0_center = _0;
        word_type _1_center = _1;
        
        // word_type _0_left = shift_val_within_warp<ShiftDirection::RIGHT>(_0_center);
        // word_type _1_left = shift_val_within_warp<ShiftDirection::RIGHT>(_1_center);

        // word_type _0_right = shift_val_within_warp<ShiftDirection::LEFT>(_0_center);
        // word_type _1_right = shift_val_within_warp<ShiftDirection::LEFT>(_1_center);

        // word_type _0_shifted = (_0_center << 1) | (_0_right >> (BITS - 1)); // 3 ops
        // word_type _1_shifted = (_1_center << 1) | (_1_right >> (BITS - 1)); // 3 ops

        word_type _0_shifted = _0_center << 1;
        word_type _1_shifted = _1_center << 1;

        add_two_bits(_0_shifted, _1_shifted, _0, _1, _2); // 7 ops

        _0_shifted = _0_center >> 1;
        _1_shifted = _1_center >> 1;
        
        // _0_shifted = (_0_center >> 1) | (_0_left << (BITS - 1)); // 3 ops
        // _1_shifted = (_1_center >> 1) | (_1_left << (BITS - 1)); // 3 ops

        add_two_bits(_0_shifted, _1_shifted, _0, _1, _2); // 7 ops

        return GOL(cc, _0, _1, _2); // 9 ops

        // total 41 😢
    }

    static __device__ __forceinline__ void add_two_bits(
        word_type _0a, word_type _1a,
        word_type& _0b, word_type& _1b, word_type& _2b
    ){
        _0b = _0a ^ _0b;
        auto _0_carry = _0a & _0b;
        
        _1b = _1b ^ _1a ^ _0_carry;
        auto _1_carry = (_1b ^ _1a) & _0_carry;

        _2b = _2b ^ _1_carry;
    }

    static __device__ __forceinline__ void add_one_bit(
        word_type a,
        word_type& _0, word_type& _1) {

        _0 = a ^ _0;
        word_type _0_carry = a & _0;

        _1 = _0_carry ^ _1;
    }

    static __host__ __device__ __forceinline__ word_type GOL(
        word_type alive, word_type _0, word_type _1, word_type _2) {

        constexpr word_type ones = ~static_cast<word_type>(0);

        auto is_3 = _0 & _1 & (_2 ^ ones);
        auto is_4 = (_0 ^ ones) & (_1 ^ ones) & _2;

        return is_3 | (is_4 & alive);

        // total 9 ops
    }
};

}

#endif