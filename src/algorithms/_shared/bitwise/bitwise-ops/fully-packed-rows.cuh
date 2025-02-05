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

    constexpr static std::size_t BITS_PER_CELL = WastefulRows<word_type>::BITS_PER_CELL;
    constexpr static std::size_t BITS = WastefulRows<word_type>::BITS;
    constexpr static std::size_t CELLS_PER_WORD = WastefulRows<word_type>::CELLS_PER_WORD;

    constexpr static word_type CELL_MASK = (static_cast<word_type>(1) << BITS_PER_CELL) - 1;

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        constexpr word_type mask_01 = constant_2<0b01>::expanded;
        constexpr word_type mask_10 = constant_2<0b10>::expanded;

        word_type neighborhoods_01 = (ct & mask_01) + (cc & mask_01) + (cb & mask_01);
        word_type neighborhoods_10 = ((ct & mask_10) >> 1) + ((cc & mask_10) >> 1) + ((cb & mask_10) >> 1);

        word_type neighborhoods_01_0011 = neighborhoods_01 & constant_4<0b0011>::expanded;
        word_type neighborhoods_01_1100 = neighborhoods_01 & constant_4<0b1100>::expanded >> 2;

        word_type neighborhoods_10_0011 = neighborhoods_10 & constant_4<0b0011>::expanded;
        word_type neighborhoods_10_1100 = neighborhoods_10 & constant_4<0b1100>::expanded >> 2;

        word_type complete_neighborhoods_01_0011 = neighborhoods_01_0011 + neighborhoods_10_0011 + (neighborhoods_10_1100 << 2);
        word_type complete_neighborhoods_01_1100 = neighborhoods_01_1100 + neighborhoods_10_1100 + (neighborhoods_01_0011 << 2);

        



        // word_type right_neighborhood =
        //     (rt & 1 << (BITS - 2)) +
        //     (rc & 1 << (BITS - 2)) +
        //     (rb & 1 << (BITS - 2));
 
        // word_type left_neighborhood =
        //     (lt >> (BITS - 1)) +
        //     (lc >> (BITS - 1)) +
        //     (lb >> (BITS - 1));

        // neighborhoods_01 += right_neighborhood;
        // neighborhoods_10 += left_neighborhood;

        return constant_4<0b1111>::expanded;
    }


    static __host__ __device__ __forceinline__  word_type compute_having_2_bits(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type neighborhoods = ct + cc + cb;
        neighborhoods += (neighborhoods >> BITS_PER_CELL) + (neighborhoods << BITS_PER_CELL);

        word_type right_neighborhoods = (rt + rc + rb) << (BITS - BITS_PER_CELL);
        word_type left_neighborhoods = (lt + lc + lb) >> (BITS - BITS_PER_CELL);

        neighborhoods += right_neighborhoods + left_neighborhoods;
        
        neighborhoods -= cc;

        word_type is_3 = (cc | neighborhoods) ^ constant_4<0b1100>::expanded;
        is_3 = (is_3 >> 2) & is_3;
        is_3 = (is_3 >> 1) & is_3;

        word_type result = is_3 & constant_4<0b0001>::expanded;

        return result;
    }
};

}

#endif