#ifndef CUDA_TILED_BIT_GRID_ON_CUDA_MODEL_HPP
#define CUDA_TILED_BIT_GRID_ON_CUDA_MODEL_HPP

#include <cstddef>
namespace algorithms {

template <typename tile_type>
struct TiledGridOnCuda {
    tile_type* input;
    tile_type* output;
    std::size_t x_size;
    std::size_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP