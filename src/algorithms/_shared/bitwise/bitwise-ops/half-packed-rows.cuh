#ifndef ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_HALF_PACKED_ROWS_CUH
#define ALGORITHMS_SHARED_BITWISE_BITWISE_OPS_HALF_PACKED_ROWS_CUH

#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../template_helpers/striped_constants.hpp"

namespace algorithms {

template <typename word_type>
struct HalfPackedWithVectorOperationsImplementation {

    template <int c>
    using constant_4 = Consts<word_type, c, 4>;

    template <int c>
    using constant_2 = Consts<word_type, c, 2>;

    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        word_type neighborhoods = ct + cc + cb;

        word_type neighborhoods_0011 = neighborhoods & constant_4<0b0011>::expanded;
        word_type neighborhoods_1100 = (neighborhoods & constant_4<0b1100>::expanded) >> 2;
        
        word_type left_neighborhoods = (lt + lc + lb) >> (BITS - 2);
        word_type right_neighborhoods = (rt + rc + rb) << (BITS - 2);

        word_type complete_neighborhoods_0011 = neighborhoods_0011 + neighborhoods_1100 + (neighborhoods_1100 << 4) + left_neighborhoods;
        word_type complete_neighborhoods_1100 = neighborhoods_1100 + neighborhoods_0011 + (neighborhoods_0011 >> 4) + (right_neighborhoods >> 2);

        word_type alive_0011 = cc & constant_4<0b0001>::expanded;
        word_type alive_1100 = (cc >> 2) & constant_4<0b0001>::expanded;

        word_type result_0011 = GOL(alive_0011, complete_neighborhoods_0011);
        word_type result_1100 = GOL(alive_1100, complete_neighborhoods_1100);

        return result_0011 | (result_1100 << 2);
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