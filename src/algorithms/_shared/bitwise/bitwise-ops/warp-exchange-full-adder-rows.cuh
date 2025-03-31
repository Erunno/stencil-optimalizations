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

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type ct, 
        word_type cc,
        word_type cb) {

        word_type _0_center = cc;
        word_type _1_center = 0;

        add_one(ct, _0_center, _1_center);
        add_one(cb, _0_center, _1_center);

        word_type _0_right = shift_val_within_warp<ShiftDirection::LEFT>(_0_center);
        word_type _1_right = shift_val_within_warp<ShiftDirection::LEFT>(_1_center);
        
        word_type _0_left = shift_val_within_warp<ShiftDirection::RIGHT>(_0_center);
        word_type _1_left = shift_val_within_warp<ShiftDirection::RIGHT>(_1_center);

        word_type r_0 = _0_center; 
        word_type r_1 = _1_center; 
        word_type r_2 = 0;
    
        word_type _0_shifted, _1_shifted;

        _0_shifted = (_0_center << 1) | (_0_left >> (BITS - 1));
        _1_shifted = (_1_center << 1) | (_1_left >> (BITS - 1));

        add_two(
            _0_shifted, _1_shifted,
            r_0, r_1, r_2);

        _0_shifted = (_0_center >> 1) | (_0_right << (BITS - 1));
        _1_shifted = (_1_center >> 1) | (_1_right << (BITS - 1));

        add_two(
            _0_shifted, _1_shifted, 
            r_0, r_1, r_2);

        return GOL(cc, r_0, r_1, r_2);
    }

    static __host__ __device__ __forceinline__ void add_one(word_type a, word_type& _0, word_type& _1) {
        word_type r_0 = a ^ _0;

        word_type _0_carry = a & _0;
        word_type r_1 = _0_carry ^ _1;

        _0 = r_0;
        _1 = r_1;
    }


    static __host__ __device__ __forceinline__ void add_two(
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

    constexpr static word_type ones = ~static_cast<word_type>(0);
    constexpr static word_type zeros = 0;

    static __host__ __device__ __forceinline__ word_type GOL(
        word_type alive, word_type _0, word_type _1, word_type _2) {

        auto is_3 = _0 & _1 & (_2 ^ ones);
        auto is_4 = (_0 ^ ones) & (_1 ^ ones) & _2;

        return is_3 | (is_4 & alive);

        // total 9 ops
    }
};

}

#endif