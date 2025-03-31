#ifndef CUDA_SIMPLE_WARP_EXCHANGE_HPP
#define CUDA_SIMPLE_WARP_EXCHANGE_HPP

#include <cstddef>
namespace algorithms {

template <typename tile_type>
struct BitGridForSimpleWarpExchange {
    tile_type* input;
    tile_type* output;
    std::size_t x_size;
    std::size_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP