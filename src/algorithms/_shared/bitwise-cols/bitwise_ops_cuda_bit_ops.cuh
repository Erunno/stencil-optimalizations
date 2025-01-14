#ifndef ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH
#define ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH

#include <cstdint>
#include "./bitwise_ops_macros.hpp"
#include <cuda_runtime.h> 

namespace algorithms {

#define POPCOUNT_16(x) __popc(x)
#define POPCOUNT_32(x) __popc(x)
#define POPCOUNT_64(x) __popcll(x)

template <typename col_type>
class CudaBitwiseOps {};

template <>
class CudaBitwiseOps<std::uint16_t> {
    using col_type = std::uint16_t;

public:
    __device__ static __forceinline__ col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {

        return __16_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};


template <>
class CudaBitwiseOps<std::uint32_t> {
    using col_type = std::uint32_t;

public:
    __device__ static __forceinline__ col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {

        return __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t> {
    using col_type = std::uint64_t;

public:
    __device__ static __forceinline__ col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {

        return __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

} // namespace algorithms
#endif // ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH