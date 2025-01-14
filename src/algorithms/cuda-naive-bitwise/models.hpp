#ifndef CUDA_NAIVE_MODELS_BIT_GRID_HPP
#define CUDA_NAIVE_MODELS_BIT_GRID_HPP

#include <cstddef>
namespace algorithms {

template <typename col_type>
struct BitGridOnCuda {
    col_type* input;
    col_type* output;
    std::size_t x_size;
    std::size_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP
