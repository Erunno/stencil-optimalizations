#ifndef CPU_GOL_OPS_FOR_AN5D_HPP
#define CPU_GOL_OPS_FOR_AN5D_HPP

#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include <cstdint>

#define POPCOUNT_16(x) __builtin_popcount(x)
#define POPCOUNT_32(x) __builtin_popcount(x)
#define POPCOUNT_64(x) __builtin_popcountll(x)

#define GOL_OP_32(lt, ct, rt, lc, cc, rc, lb, cb, rb)                                                                  \
    __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)
#define GOL_OP_64(lt, ct, rt, lc, cc, rc, lb, cb, rb)                                                                  \
    __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)

#endif // GOL_OPS_FOR_AN5D_HPP
