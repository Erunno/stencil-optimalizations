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