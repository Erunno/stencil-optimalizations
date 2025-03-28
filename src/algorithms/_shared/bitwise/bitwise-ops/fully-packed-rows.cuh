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

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type _0 = cc;
        word_type _1 = 0;
        word_type _2 = 0;
        word_type _3 = 0;

        add<2>(ct, _0, _1, _2, _3);
        add<2>(cb, _0, _1, _2, _3);

        add_two<3>(_0 << 1, _1 << 1, _2 << 1, _3 << 1, _0, _1, _2, _3);
        add_two<4>(_0 >> 1, _1 >> 1, _2 >> 1, _3 >> 1, _0, _1, _2, _3);

        return GOL(cc, _0, _1, _2, _3);
    }

    constexpr static word_type ones = ~static_cast<word_type>(0);
    constexpr static word_type zeros = 0;

    static __host__ __device__ __forceinline__ word_type GOL(
        word_type alive, word_type _0, word_type _1, word_type _2, word_type _3) {

        minus(alive, _0, _1, _2, _3);
        _0 = _0 | alive;
        return (_0 ^ ones) & (_1 ^ ones) & (_2 ^ zeros) & (_3 ^ zeros); 

        // neighborhood -= alive;

        // word_type is_3 = (alive | neighborhood) ^ constant_4<0b1100>::expanded;
        // is_3 = (is_3 >> 2) & is_3;
        // is_3 = (is_3 >> 1) & is_3;

        // word_type result = is_3 & constant_4<0b0001>::expanded;

        // return result;
    }

    static __host__ __device__ __forceinline__ void minus(word_type a, word_type& _0, word_type& _1, word_type& _2, word_type& _3) {
        add_two<3>(a, a, a, a, _0, _1, _2, _3);
    }

    template <int max_used_bit>
    static __host__ __device__ __forceinline__ void add(word_type a, word_type& _0, word_type& _1, word_type& _2, word_type& _3) {
        _0 = a ^ _0;

        if constexpr (max_used_bit == 0) return;

        word_type _0_carry = a & _0;
        _1 = _0_carry ^ _1;

        if constexpr (max_used_bit == 1) return;

        word_type _1_carry = _0_carry & _1;
        _2 = _1_carry ^ _2;

        if constexpr (max_used_bit == 2) return;

        word_type _2_carry = _1_carry & _2;
        _3 = _2_carry ^ _3;
    }

    template <int max_used_bit>
    static __host__ __device__ __forceinline__ void add_two(
        word_type _0a, word_type _1a, word_type _2a, word_type _3a,
        word_type& _0, word_type& _1, word_type& _2, word_type& _3
    ){
        _0 = _0a ^ _0;
        
        if constexpr (max_used_bit == 0) return;
        
        auto _0_carry = _0a & _0;
        _1 = _1 ^ _1a ^ _0_carry;

        if constexpr (max_used_bit == 1) return;

        auto _1_carry = (_1 ^ _1a) & _0_carry;
        _2 = _2 ^ _2a ^ _1_carry;

        if constexpr (max_used_bit == 2) return;

        auto _2_carry = (_2 ^ _2a) & _1_carry;
        _3 = _3 ^ _3a ^ _2_carry;
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