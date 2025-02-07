#ifndef ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_ADDER_CUH
#define ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_ADDER_CUH

#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../template_helpers/striped_constants.hpp"

namespace algorithms {

template <typename word_type>
struct AdderOperationsImplementation {

    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type _0 = cc;
        word_type _1 = 0;

        add(ct, _0, _1); // 3 ops
        add(cb, _0, _1); // 3 ops

        word_type _0_right_nei, _1_right_nei;
        word_type _0_left_nei, _1_left_nei;
        
        load_right_neighborhood(rt, rc, rb, _0_right_nei, _1_right_nei); // 5 ops + 3 ifs (+ 3 ops if 'ifs' are not used)
        load_left_neighborhood(lt, lc, lb, _0_left_nei, _1_left_nei); // 5 ops + 3 ifs (+ 2 ops if 'ifs' are not used)

        word_type r_0 = _0; 
        word_type r_1 = _1; 
        word_type r_2 = 0;
    
        constexpr bool _2a_is_zero = false;
        
        add_two<_2a_is_zero>(
            (_0 << 1) | _0_left_nei, (_1 << 1) | _1_left_nei, 0,
            r_0, r_1, r_2); // 8 ops + 4 ops

        add_two<_2a_is_zero>(
            (_0 >> 1) | _0_right_nei, (_1 >> 1) | _1_right_nei, 0, 
            r_0, r_1, r_2); // 8 ops + 4 ops

        return GOL(cc, r_0, r_1, r_2); // 9 ops

        // total 49 ops + 6 ifs (+ 5 ops if 'ifs' are not used)
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

    static __host__ __device__ __forceinline__ void add(word_type a, word_type& _0, word_type& _1) {
        auto r_0 = a ^ _0;
        word_type _0_carry = a & _0;

        auto r_1 = _0_carry ^ _1;

        _0 = r_0;
        _1 = r_1;

        // used 3 ops
    }


    template <bool _2a_is_zero = false>
    static __host__ __device__ __forceinline__ void add_two(
        word_type _0a, word_type _1a, word_type _2a,
        word_type& _0, word_type& _1, word_type& _2
    ){
        auto _1_xor_1a = _1 ^ _1a;

        auto r_0 = _0a ^ _0;

        auto _0_carry = _0a & _0;
        auto r_1 = _1_xor_1a ^ _0_carry;

        auto _1_carry = (_1 & _1a) | (_0_carry & (_1_xor_1a));

        if constexpr (_2a_is_zero) {
            _2 = _2 ^ _1_carry;
            // used 8 ops
        } else {
            _2 = _2 ^ _2a ^ _1_carry;
            // used 9 ops
        }

        _0 = r_0;
        _1 = r_1;
    }

    static __host__ __device__ __forceinline__  void load_right_neighborhood(
        word_type rt, word_type rc, word_type rb, 
        word_type& _0, word_type& _1) {

        constexpr word_type ONE = static_cast<word_type>(1) << (BITS - 1);

        auto count = (rt & 1) + (rc & 1) + (rb & 1);
        set_words<0, ONE>(count, _0, _1);

        // used 5 ops + 3 ifs (+ 3 ops if 'ifs' are not used)
    }

    static __host__ __device__ __forceinline__  void load_left_neighborhood(
        word_type lt, word_type lc, word_type lb, 
        word_type& _0, word_type& _1) {

        constexpr word_type SHIFT = BITS - 1;

        auto count = (lt >> SHIFT) + (lc >> SHIFT) + (lb >> SHIFT);
        set_words<0, 1>(count, _0, _1);

        // used 5 ops + 3 ifs (+ 2 ops if 'ifs' are not used)
    }

    template <word_type zero, word_type one>
    static __host__ __device__ __forceinline__  void set_words(
        word_type count, 
        word_type& _0, word_type& _1) {

        if (count == 0) {
            _0 = zero;
            _1 = zero;
        } else if (count == 1) {
            _0 = one;
            _1 = zero;
        } else if (count == 2) {
            _0 = zero;
            _1 = one;
        } else {
            _0 = one;
            _1 = one;
        }

        // variant without ifs (just a bit slower)

        // if constexpr (one == 1) {
        //     _0 = count & 1;
        //     _1 = count >> 1;
        // } 
        // else {
        //     _0 = count << (BITS - 1);
        //     _1 = (count << (BITS - 2)) & one;
        // }

    }
};

}

#endif