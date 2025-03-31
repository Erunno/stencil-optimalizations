#ifndef ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
#define ALGORITHMS_WARP_EXCHANGE_FULL_ADDER_ON_ROWS
    
#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../template_helpers/striped_constants.hpp"

namespace algorithms {

template <typename word_type>
struct WarpExchangeFullAdderOnRows {

    
    constexpr static int BITS = sizeof(word_type) * 8;

    static __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        auto o_lt = lt;
        auto o_ct = ct;
        auto o_rt = rt;

        auto o_lc = lc;
        auto o_cc = cc;
        auto o_rc = rc;

        auto o_lb = lb;
        auto o_cb = cb;
        auto o_rb = rb;

        lt = o_lt;
        ct = o_lc;
        rt = o_lb;

        lc = o_ct;
        cc = o_cc;
        rc = o_cb;

        lb = o_rt;
        cb = o_rc;
        rb = o_rb;
        
        // the top-left neighbors of the center cell:
        const word_type A = (lc << 1) | (lt >> (BITS - 1));
        // the top neighbors of the center cell:
        const word_type B = (cc << 1) | (ct >> (BITS - 1));
        // the top-right neighbors of the center cell:
        const word_type C = (rc << 1) | (rt >> (BITS - 1));
        // the right neighbors of the center cell:
        const word_type D = rc;
        // the bottom-right neighbors of the center cell:
        const word_type E = (rc >> 1) | (rb << (BITS - 1));
        // the bottom neighbors of the center cell:
        const word_type F = (cc >> 1) | (cb << (BITS - 1));
        // the bottom-left neighbors of the center cell:
        const word_type G = (lc >> 1) | (lb << (BITS - 1));
        // the left neighbors of the center cell:
        const word_type H = lc;
        const word_type I = cc;

        return compute_center_word_fujita(A, B, C, H, I, D, G, F, E);
    }

    static __device__ __forceinline__ word_type compute_center_word_fujita(
        word_type A, word_type B, word_type C, 
        word_type H, word_type I, word_type D,
        word_type G, word_type F, word_type E) {
        // 1.
        const word_type AB_1 = A & B;
        const word_type AB_0 = A ^ B;
        // 2.
        const word_type CD_1 = C & D;
        const word_type CD_0 = C ^ D;
        // 3.
        const word_type EF_1 = E & F;
        const word_type EF_0 = E ^ F;
        // 4.
        const word_type GH_1 = G & H;
        const word_type GH_0 = G ^ H;
        // 5.
        const word_type AD_0 = AB_0 ^ CD_0;
        // 6.
        const word_type AD_1 = AB_1 ^ CD_1 ^ (AB_0 & CD_0);
        // 7.
        const word_type AD_2 = AB_1 & CD_1;
        // 8.
        const word_type EH_0 = EF_0 ^ GH_0;
        // 9.
        const word_type EH_1 = EF_1 ^ GH_1 ^ (EF_0 & GH_0);
        // 10.
        const word_type EH_2 = EF_1 & GH_1;
        // 11.
        const word_type AH_0 = AD_0 ^ EH_0;
        // 12.
        const word_type X = AD_0 & EH_0;
        // 13.
        const word_type Y = AD_1 ^ EH_1;
        // 14.
        const word_type AH_1 = X ^ Y;
        // 15.
        const word_type AH_23 = AD_2 | EH_2 | (AD_1 & EH_1) | (X & Y);
        // 17. neither of the 2 most significant bits is set and the second least significant bit is set
        const word_type Z = ~AH_23 & AH_1;
        // 18. (two neighbors) the least significant bit is not set and Z
        const word_type I_2 = ~AH_0 & Z;
        // 19. (three neighbors) the least significant bit is set and Z
        const word_type I_3 = AH_0 & Z;
        // 20.
        return (I & I_2) | I_3;
    }

};

}

#endif