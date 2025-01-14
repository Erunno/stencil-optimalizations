#ifndef CUDA_GOL_OPS_FOR_AN5D_HPP
#define CUDA_GOL_OPS_FOR_AN5D_HPP

#include "../_shared/bitwise-cols/bitwise_ops_macros.hpp"
#include <cstdint>

#define POPCOUNT_16(x) __popc(x)
#define POPCOUNT_32(x) __popc(x)
#define POPCOUNT_64(x) __popcll(x)

#define GOL_OP_32(lt, ct, rt, lc, cc, rc, lb, cb, rb)                                                                  \
    __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)
#define GOL_OP_64(lt, ct, rt, lc, cc, rc, lb, cb, rb)                                                                  \
    __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb)

#include "an5d_cuda_timer.hpp"
#define STENCILBENCH
#define SB_START_INSTRUMENTS algorithms::An5dCudaTimer::start();
#define SB_STOP_INSTRUMENTS algorithms::An5dCudaTimer::stop();

#endif // GOL_OPS_FOR_AN5D_HPP
