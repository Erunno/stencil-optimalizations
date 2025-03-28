#ifndef ALGORITHMS_TILED_FULL_ADDER
#define ALGORITHMS_TILED_FULL_ADDER
    
#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include <cuda_runtime.h>
#include "../../template_helpers/striped_constants.hpp"

namespace algorithms {

template <typename word_type>
struct TiledFullAdder {

    constexpr static int BITS = sizeof(word_type) * 8;
    // For tiles like 8x8 in a 64-bit word
    constexpr static int X_BITS = 8;  // Width of tile
    constexpr static int Y_BITS = 8;  // Height of tile

    // Simplified version - focusing on correct bit arrangement
    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        constexpr word_type BOTTOM_LINE = 0b1111'1111;
        constexpr word_type TOP_LINE = BOTTOM_LINE << (BITS - X_BITS);
        constexpr word_type RIGHT_BORDER = 0x80'80'80'80'80'80'80'80LLU; 
        constexpr word_type LEFT_BORDER = 0x01'01'01'01'01'01'01'01LLU;

        word_type small_top = ct << (BITS - X_BITS);
        word_type big_bottom = cc >> X_BITS;  

        word_type A = small_top | big_bottom;
        word_type B = (small_top << 1) | (big_bottom << 1);
        word_type C = (small_top >> 1) | (big_bottom >> 1);

        word_type big_top = cc << X_BITS;
        word_type small_bottom = cb >> (BITS - X_BITS);

        word_type D = big_top | small_bottom;
        word_type E = (big_top << 1) | (small_bottom << 1);
        word_type F = (big_top >> 1) | (small_bottom >> 1);

        word_type big_right = (cc >> 1) & ~LEFT_BORDER;
        word_type small_left = (lc << 7) & LEFT_BORDER;

        word_type big_left = (cc << 1) & ~RIGHT_BORDER;
        word_type small_right = (rc >> 1) & RIGHT_BORDER;

        word_type G = big_right | small_left;
        word_type H = big_left | small_right;

        word_type I = cc;
        
        return compute_center_word_full_adder(A, B, C, H, I, D, G, F, E);
    }

    static __host__ __device__ __forceinline__ word_type compute_center_word_full_adder(
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