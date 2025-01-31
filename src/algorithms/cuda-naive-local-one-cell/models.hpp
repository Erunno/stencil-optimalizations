#ifndef ALGORITHMS_CUDA_LOCAL_ONE_CELL_MODELS_HPP
#define ALGORITHMS_CUDA_LOCAL_ONE_CELL_MODELS_HPP

#include <cstddef>
namespace algorithms::cuda_local_one_cell {

template <typename change_state_store_type>
struct ChangeStateStore {
    change_state_store_type* last;
    change_state_store_type* current;
};

template <typename word_type, typename change_state_store_type>
struct BitGridWithChangeInfo {
    constexpr static std::size_t BITS = sizeof(word_type) * 8;

    word_type* input;
    word_type* output;

    std::size_t x_size;
    std::size_t y_size;

    ChangeStateStore<change_state_store_type> change_state_store;
};

struct StateStoreInfo {

    static constexpr std::size_t CACHE_SIZE_X = 3;
    static constexpr std::size_t CACHE_SIZE_Y = 3;

    static constexpr std::size_t CACHE_SIZE = CACHE_SIZE_X * CACHE_SIZE_Y; 
};

}

#endif // ALGORITHMS_CUDA_LOCAL_ONE_CELL_MODELS_HPP