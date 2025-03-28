#ifndef GOL_CPU_BITWISE_COLS_MACRO_HPP
#define GOL_CPU_BITWISE_COLS_MACRO_HPP

#include "./macro-cols.hpp"
#include "./macro-tiles.hpp"
#include <cstdint>
#include <iostream>
#include "../bit_modes.hpp"
#include "./wasteful-rows.cuh"
#include "./fully-packed-rows.cuh"
#include "./half-packed-rows.cuh"
#include "./fujita.cuh"
#include "./adder.cuh"
#include "./tiled-full-adder.cuh"

namespace algorithms {

#undef POPCOUNT_16
#undef POPCOUNT_32
#undef POPCOUNT_64

#define POPCOUNT_16(x) __builtin_popcount(x)
#define POPCOUNT_32(x) __builtin_popcount(x)
#define POPCOUNT_64(x) __builtin_popcountll(x)

template <typename word_type>
class MacroColsOperations {};

template <typename word_type>
class MacroTilesOperations {};

template <>
class MacroColsOperations<std::uint16_t> {
  public:
    using word_type = std::uint16_t;
    using bit_grid_mode = BitColumnsMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
    
        return __16_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroColsOperations<std::uint32_t> {
  public:
    using word_type = std::uint32_t;
    using bit_grid_mode = BitColumnsMode;

    // clang-format off

    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
        
        return __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroColsOperations<std::uint64_t> {
  public:
    using word_type = std::uint64_t;
    using bit_grid_mode = BitColumnsMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};


template <>
class MacroTilesOperations<std::uint16_t> {
  public:
    using word_type = std::uint16_t;
    using bit_grid_mode = BitTileMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
    
        return __16_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroTilesOperations<std::uint32_t> {
  public:
    using word_type = std::uint32_t;
    using bit_grid_mode = BitTileMode;

    // clang-format off

    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
        
        return __32_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroTilesOperations<std::uint64_t> {
  public:
    using word_type = std::uint64_t;
    using bit_grid_mode = BitTileMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class WastefulRowsOps {
  public:
    using bit_grid_mode = BitWastefulRowsMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return WastefulRowsImplantation<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class HalfPackedRowsWithVectorOps {
  public:
    using bit_grid_mode = HalfPackedRowsMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return HalfPackedWithVectorOperationsImplementation<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class FullyPackedRowsWithVectorOps {
  public:
    using bit_grid_mode = FullyPackedRowsMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return FullyPackedWithVectorOperationsImplementation<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class AdderOps {
  public:
    using bit_grid_mode = AdderMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return AdderOperationsImplementation<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class FujitaOps {
  public:
    using bit_grid_mode = FujitaMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return FujitaImplantation<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <typename word_type>
class TiledFullAdderOps {
  public:
    using bit_grid_mode = TiledFullAdderMode;

    // clang-format off
    static  word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
            
        return TiledFullAdder<word_type>::compute_center_word(
            lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};



} // namespace algorithms

#endif // GOL_CPU_BITWISE_COLS_HPP