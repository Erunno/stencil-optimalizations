#ifndef CUDA_NAIVE_MODELS_HPP
#define CUDA_NAIVE_MODELS_HPP

#include <cstddef>
namespace algorithms {

struct NaiveGridOnCuda {
    char* input;
    char* output;
    std::size_t x_size;
    std::size_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP
