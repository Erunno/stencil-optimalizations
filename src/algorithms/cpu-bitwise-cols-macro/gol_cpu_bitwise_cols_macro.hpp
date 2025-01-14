#ifndef GOL_CPU_BITWISE_COLS_MACRO_HPP
#define GOL_CPU_BITWISE_COLS_MACRO_HPP

#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include <cstdint>

namespace algorithms {

#define POPCOUNT_16(x) __builtin_popcount(x)
#define POPCOUNT_32(x) __builtin_popcount(x)
#define POPCOUNT_64(x) __builtin_popcountll(x)

template <typename col_type>
class MacroBitOperations {};

template <>
class MacroBitOperations<std::uint16_t> {
  public:
    using col_type = std::uint16_t;

    // clang-format off
    static  col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {
    
        return __16_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroBitOperations<std::uint32_t> {
  public:
    using col_type = std::uint32_t;

    // clang-format off

    static  col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {
        
        return __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

template <>
class MacroBitOperations<std::uint64_t> {
  public:
    using col_type = std::uint64_t;

    // clang-format off
    static  col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {

        return __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }

    // clang-format on
};

} // namespace algorithms

#endif // GOL_CPU_BITWISE_COLS_HPP
