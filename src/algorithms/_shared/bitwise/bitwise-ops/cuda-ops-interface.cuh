#ifndef ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH
#define ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH

#include <cstdint>
#include "./macro-cols.hpp"
#include "./macro-tiles.hpp"
#include <cuda_runtime.h> 
#include "../bit_modes.hpp"
#include "./wasteful-rows.cuh"
#include "./fully-packed-rows.cuh"
#include "./half-packed-rows.cuh"
#include "./fujita.cuh"
#include "./adder.cuh"
#include "./tiled-full-adder.cuh"
#include "./warp-exchange-full-adder-rows.cuh"

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

template <typename word_type>
class CudaBitwiseOps<word_type, BitWastefulRowsMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return WastefulRowsImplantation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, HalfPackedRowsMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return HalfPackedWithVectorOperationsImplementation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, FullyPackedRowsMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return FullyPackedWithVectorOperationsImplementation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, AdderMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return AdderOperationsImplementation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, FujitaMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return FujitaImplantation<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, TiledFullAdderMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {

        return TiledFullAdder<word_type>::compute_center_word(lt, ct, rt, lc, cc, rc, lb, cb, rb);
    }
};

template <typename word_type>
class CudaBitwiseOps<word_type, WarpExchangeFullAdderOnRowsMode> {
public:
    __device__ static __forceinline__ word_type compute_center_word(
        word_type ct, 
        word_type cc,
        word_type cb) {

        return WarpExchangeFullAdderOnRows<word_type>::compute_center_word(ct, cc, cb);
    }
};

} // namespace algorithms
#endif // ALGORITHMS_BITWISE_OPS_CUDA_BIT_OPS_CUH