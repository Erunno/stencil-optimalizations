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
    constexpr static int X_BITS = 8;
    constexpr static int Y_BITS = 8;

    static __host__ __device__ __forceinline__  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        // Masks

        constexpr word_type BOTTOM_LINE = static_cast<word_type>(0b1111'1111);
        constexpr word_type TOP_LINE = BOTTOM_LINE << (BITS - X_BITS);
        constexpr word_type LEFT_BORDER = static_cast<word_type>(0x80'80'80'80'80'80'80'80LLU); 
        constexpr word_type RIGHT_BORDER = static_cast<word_type>(0x01'01'01'01'01'01'01'01LLU);

        constexpr word_type A_cc_mask = ~TOP_LINE & ~LEFT_BORDER;
        constexpr word_type A_ct_mask = (TOP_LINE >> 1) & TOP_LINE;
        constexpr word_type A_lt_mask = TOP_LINE << (X_BITS - 1);
        constexpr word_type A_lc_mask = (LEFT_BORDER >> X_BITS);

        constexpr word_type B_cc_mask = ~TOP_LINE;
        constexpr word_type B_ct_mask = TOP_LINE;

        constexpr word_type C_cc_mask = ~TOP_LINE & ~RIGHT_BORDER;
        constexpr word_type C_ct_mask = TOP_LINE << 1;
        constexpr word_type C_rc_mask = RIGHT_BORDER >> X_BITS;
        constexpr word_type C_rt_mask = (TOP_LINE >> (X_BITS - 1)) & TOP_LINE;

        constexpr word_type D_cc_mask = ~LEFT_BORDER;
        constexpr word_type D_lc_mask = LEFT_BORDER;
        
        constexpr word_type E_cc_mask = ~RIGHT_BORDER;
        constexpr word_type E_rc_mask = RIGHT_BORDER;

        constexpr word_type F_cc_mask = ~LEFT_BORDER & ~ BOTTOM_LINE;
        constexpr word_type F_lc_mask = LEFT_BORDER << X_BITS;
        constexpr word_type F_lb_mask = 1 << (X_BITS - 1);
        constexpr word_type F_cb_mask = BOTTOM_LINE >> 1;
        
        constexpr word_type G_cc_mask = ~BOTTOM_LINE;
        constexpr word_type G_cb_mask = BOTTOM_LINE;
        
        constexpr word_type H_cc_mask = ~RIGHT_BORDER & ~BOTTOM_LINE;
        constexpr word_type H_rc_mask = RIGHT_BORDER << X_BITS;
        constexpr word_type H_cb_mask = (BOTTOM_LINE << 1) & BOTTOM_LINE;
        constexpr word_type H_rb_mask = 1;

        // Shifts

        constexpr int A_cc_R_shift = X_BITS + 1;
        constexpr int A_ct_L_shift = (Y_BITS - 1) * X_BITS - 1;
        constexpr int A_lc_R_shift = 1;
        constexpr int A_lt_L_shift = X_BITS * Y_BITS - 1;

        constexpr int B_cc_R_shift = X_BITS;
        constexpr int B_ct_L_shift = (Y_BITS - 1) * X_BITS;
        
        constexpr int C_cc_R_shift = X_BITS - 1;
        constexpr int C_ct_L_shift = (Y_BITS - 1) * X_BITS + 1;
        constexpr int C_rc_R_shift = 2 * X_BITS - 1;
        constexpr int C_rt_L_shift = X_BITS * (Y_BITS - 2) + 1;
        
        constexpr int D_cc_R_shift = 1;
        constexpr int D_lc_L_shift = X_BITS - 1;
        
        constexpr int E_cc_L_shift = 1;
        constexpr int E_rc_R_shift = X_BITS - 1;
        
        constexpr int F_cc_L_shift = X_BITS - 1;
        constexpr int F_cb_R_shift = (Y_BITS - 1) * X_BITS + 1;
        constexpr int F_lc_L_shift = 2 * X_BITS - 1;
        constexpr int F_lb_R_shift = X_BITS * (Y_BITS - 2) + 1;
        
        constexpr int G_cc_L_shift = X_BITS;
        constexpr int G_cb_R_shift = (Y_BITS - 1) * X_BITS;
        
        constexpr int H_cc_L_shift = X_BITS + 1;
        constexpr int H_cb_R_shift = (Y_BITS - 1) * X_BITS - 1;
        constexpr int H_rc_L_shift = 1;
        constexpr int H_rb_R_shift = X_BITS * Y_BITS - 1;

        
        word_type A =
            ((cc >> A_cc_R_shift) & A_cc_mask) |
            ((ct << A_ct_L_shift) & A_ct_mask) |
            ((lc >> A_lc_R_shift) & A_lc_mask) |
            ((lt << A_lt_L_shift) & A_lt_mask);
        word_type B =
            ((cc >> B_cc_R_shift) & B_cc_mask) |
            ((ct << B_ct_L_shift) & B_ct_mask);
        word_type C =
            ((cc >> C_cc_R_shift) & C_cc_mask) |
            ((ct << C_ct_L_shift) & C_ct_mask) |
            ((rc >> C_rc_R_shift) & C_rc_mask) |
            ((rt << C_rt_L_shift) & C_rt_mask);
        word_type D =
            ((cc >> D_cc_R_shift) & D_cc_mask) |
            ((lc << D_lc_L_shift) & D_lc_mask);
        word_type E =
            ((cc << E_cc_L_shift) & E_cc_mask) |
            ((rc >> E_rc_R_shift) & E_rc_mask);
        word_type F =
            ((cc << F_cc_L_shift) & F_cc_mask) |
            ((cb >> F_cb_R_shift) & F_cb_mask) |
            ((lc << F_lc_L_shift) & F_lc_mask) |
            ((lb >> F_lb_R_shift) & F_lb_mask);
        word_type G =
            ((cc << G_cc_L_shift) & G_cc_mask) |
            ((cb >> G_cb_R_shift) & G_cb_mask);
        word_type H =
            ((cc << H_cc_L_shift) & H_cc_mask) |
            ((cb >> H_cb_R_shift) & H_cb_mask) |
            ((rc << H_rc_L_shift) & H_rc_mask) |
            ((rb >> H_rb_R_shift) & H_rb_mask);

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