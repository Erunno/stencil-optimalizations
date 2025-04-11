#ifndef CUDA_ROWED_BIT_GRID_ON_CUDA_MODEL_HPP
#define CUDA_ROWED_BIT_GRID_ON_CUDA_MODEL_HPP

#include <cstddef>
#include <cstdint>
namespace algorithms {

template <typename row_type>
struct RowedGridOnCudaForTemporal {
    row_type* input;
    row_type* output;
    std::int64_t x_size;
    std::int64_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP