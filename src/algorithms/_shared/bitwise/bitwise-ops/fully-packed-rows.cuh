#ifndef ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_FULLY_PACKED_ROWS_CUH
#define ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_FULLY_PACKED_ROWS_CUH

#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../template_helpers/striped_constants.hpp"

namespace algorithms {

template <typename word_type>
struct FullyPackedWithVectorOperationsImplementation {

    template <int c>
    using constant_4 = Consts<word_type, c, 4>;

    template <int c>
    using constant_2 = Consts<word_type, c, 2>;

    constexpr static std::size_t BITS = sizeof(word_type) * 8;
    constexpr static word_type mask_01 = constant_2<0b01>::expanded;
    constexpr static word_type mask_0001 = constant_4<0b0001>::expanded;
    constexpr static word_type mask_0011 = constant_4<0b0011>::expanded;
    constexpr static word_type mask_1100 = constant_4<0b1100>::expanded;

    constexpr static int USE_2_BITS = 2;
    constexpr static int USE_3_BITS = 3;


    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type _0 = cc;
        word_type _1 = 0;
        word_type _2 = 0;

        add<USE_2_BITS>(ct, _0, _1, _2);
        add<USE_2_BITS>(cb, _0, _1, _2);

        word_type _0_right_nei, _1_right_nei;
        word_type _0_left_nei, _1_left_nei;
        
        load_right_neighborhood(rt, rc, rb, _0_right_nei, _1_right_nei);
        load_left_neighborhood(lt, lc, lb, _0_left_nei, _1_left_nei);

        word_type r_0 = _0; 
        word_type r_1 = _1; 
        word_type r_2 = _2;
        
        add_two<USE_3_BITS>(
            (_0 << 1) | _0_left_nei, (_1 << 1) | _1_left_nei, _2 << 1, 
            r_0, r_1, r_2);
        add_two<USE_3_BITS>(
            (_0 >> 1) | _0_right_nei, (_1 >> 1) | _1_right_nei, _2 >> 1, 
            r_0, r_1, r_2);

        return GOL(cc, r_0, r_1, r_2);
    }

    constexpr static word_type ones = ~static_cast<word_type>(0);
    constexpr static word_type zeros = 0;

    static __host__ __device__ __forceinline__ word_type GOL(
        word_type alive, word_type _0, word_type _1, word_type _2) {

        minus(alive, _0, _1, _2);
        _0 = _0 | alive;
        return _0 & _1 & (_2 ^ ones);  
    }

    static __host__ __device__ __forceinline__ void minus(word_type a, word_type& _0, word_type& _1, word_type& _2) {
        add_two<USE_3_BITS>(a, a, a, _0, _1, _2);
    }

    template <int bits = USE_3_BITS>
    static __host__ __device__ __forceinline__ void add(word_type a, word_type& _0, word_type& _1, word_type& _2) {
        auto r_0 = a ^ _0;

        if constexpr (bits == 1) {
            _0 = r_0;
            return;
        }

        word_type _0_carry = a & _0;
        auto r_1 = _0_carry ^ _1;

        if constexpr (bits == 2) {
            _0 = r_0;
            _1 = r_1;
            return;
        }

        word_type _1_carry = _0_carry & _1;
        auto r_2 = _1_carry ^ _2;

        _0 = r_0;
        _1 = r_1;
        _2 = r_2;
    }


    template <int bits = USE_3_BITS>
    static __host__ __device__ __forceinline__ void add_two(
        word_type _0a, word_type _1a, word_type _2a,
        word_type& _0, word_type& _1, word_type& _2
    ){
        auto r_0 = _0a ^ _0;

        if constexpr (bits == 1) {
            _0 = r_0;
            return;
        }

        auto _0_carry = _0a & _0;
        auto r_1 = _1 ^ _1a ^ _0_carry;

        if constexpr (bits == 2) {
            _0 = r_0;
            _1 = r_1;
            return;
        }

        auto _1_carry = (_1 & _1a) | (_0_carry & (_1 ^ _1a));
        auto r_2 = _2 ^ _2a ^ _1_carry;

        _0 = r_0;
        _1 = r_1;
        _2 = r_2;
    }

    static __host__ __device__ __forceinline__  void load_right_neighborhood(
        word_type rt, word_type rc, word_type rb, 
        word_type& _0, word_type& _1) {

        constexpr word_type ONE = static_cast<word_type>(1) << (BITS - 1);

        auto count = (rt & 1) + (rc & 1) + (rb & 1);
        set_words<0, ONE>(count, _0, _1);
    }

    static __host__ __device__ __forceinline__  void load_left_neighborhood(
        word_type lt, word_type lc, word_type lb, 
        word_type& _0, word_type& _1) {

        constexpr word_type SHIFT = BITS - 1;

        auto count = (lt >> SHIFT) + (lc >> SHIFT) + (lb >> SHIFT);
        set_words<0, 1>(count, _0, _1);
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
    }





    static __host__ __device__ __forceinline__  word_type compute_center_word_split_to_four(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type neighborhoods_01 = (ct & mask_01) + (cc & mask_01) + (cb & mask_01);
        word_type neighborhoods_10 = ((ct >> 1) & mask_01) + ((cc >> 1) & mask_01) + ((cb >> 1) & mask_01);

        word_type left_neighborhood = (lt >> (BITS - 1)) + (lc >> (BITS - 1)) + (lb >> (BITS - 1));
        word_type right_neighborhood = ((rt & mask_0001) + (rc & mask_0001) + (rb & mask_0001)) << (BITS - 4);
    
        word_type neighborhoods_01_0011 = neighborhoods_01 & mask_0011;
        word_type neighborhoods_01_1100 = neighborhoods_01 & mask_1100;
        word_type neighborhoods_10_0011 = neighborhoods_10 & mask_0011;
        word_type neighborhoods_10_1100 = neighborhoods_10 & mask_1100;

        word_type complete_neighborhoods_01_0011 =  neighborhoods_01_0011       +  neighborhoods_10_0011       + (neighborhoods_10_1100 << 2) + left_neighborhood;
        word_type complete_neighborhoods_01_1100 = (neighborhoods_01_1100 >> 2) + (neighborhoods_10_1100 >> 2) +  neighborhoods_10_0011;
        word_type complete_neighborhoods_10_0011 =  neighborhoods_10_0011       +  neighborhoods_01_0011       + (neighborhoods_01_1100 >> 2);
        word_type complete_neighborhoods_10_1100 = (neighborhoods_10_1100 >> 2) + (neighborhoods_01_1100 >> 2) + (neighborhoods_01_0011 >> 4) + right_neighborhood;

        word_type result_01_0011 = GOL( cc       & mask_0001, complete_neighborhoods_01_0011);
        word_type result_01_1100 = GOL((cc >> 2) & mask_0001, complete_neighborhoods_01_1100);
        word_type result_10_0011 = GOL((cc >> 1) & mask_0001, complete_neighborhoods_10_0011);
        word_type result_10_1100 = GOL((cc >> 3) & mask_0001, complete_neighborhoods_10_1100);
        
        return result_01_0011 | (result_01_1100 << 2) | (result_10_0011 << 1) | (result_10_1100 << 3);
    }

    static __host__ __device__ __forceinline__ word_type GOL(
        word_type alive, word_type neighborhood) {
        
        neighborhood -= alive;

        word_type is_3 = (alive | neighborhood) ^ constant_4<0b1100>::expanded;
        is_3 = (is_3 >> 2) & is_3;
        is_3 = (is_3 >> 1) & is_3;

        word_type result = is_3 & constant_4<0b0001>::expanded;

        return result;
    }
};

}

#endif