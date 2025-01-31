#ifndef ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH
#define ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH

#include <cstdint>
#include "./macro-cols.hpp"
#include "./macro-tiles.hpp"
#include <cuda_runtime.h> 
#include "../bit_modes.hpp"

namespace algorithms {

#undef POPCOUNT_16
#undef POPCOUNT_32
#undef POPCOUNT_64

#define POPCOUNT_16(x) __popc(x)
#define POPCOUNT_32(x) __popc(x)
#define POPCOUNT_64(x) __popcll(x)

template <typename word_type, typename bit_grid_model>
class CudaBitwiseOps {};

template <>
class CudaBitwiseOps<std::uint16_t, BitColumnsMode> {
    using word_type = std::uint16_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __16_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};


template <>
class CudaBitwiseOps<std::uint32_t, BitColumnsMode> {
    using word_type = std::uint32_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __32_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t, BitColumnsMode> {
    using word_type = std::uint64_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_COL_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint16_t, BitTileMode> {
    using word_type = std::uint16_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __16_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};


template <>
class CudaBitwiseOps<std::uint32_t, BitTileMode> {
    using word_type = std::uint32_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __32_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <>
class CudaBitwiseOps<std::uint64_t, BitTileMode> {
    using word_type = std::uint64_t;

public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return __64_BITS__GOL_BITWISE_TILES_COMPUTE(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

} // namespace algorithms
#endif // ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH