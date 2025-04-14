#ifndef BORDER_POLICIES_CUH
#define BORDER_POLICIES_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace border_policies {

struct border_policy_independent {
    template <typename idx_t, typename size_type>
    static __device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, size_type x_size) {
        return y * x_size + x;
    }
};

struct zeros_border {
    template <typename idx_t, typename grid_type>
    static __device__ __forceinline__ auto load(idx_t x, idx_t y, grid_type data) -> decltype(data.input[0] + 0) {
        if (x < 0 || y < 0 || x >= data.x_size || y >= data.y_size)
            return 0;

        return data.input[border_policy_independent::get_idx(x, y, data.x_size)];
    }

    template <typename idx_t, typename grid_type>
    __device__ __forceinline__ idx_t get_idx(idx_t x, idx_t y, grid_type data) {
        return y * data.x_size + x;
    }
};

struct wrap_around_border {
    template <typename idx_t, typename grid_type>
    static __device__ __forceinline__ auto load(idx_t x, idx_t y, grid_type data) -> decltype(data.input[0] + 0) {
        if (x < 0)
            x += data.x_size;
        else if (x >= data.x_size)
            x -= data.x_size;

        if (y < 0)
            y += data.y_size;
        else if (y >= data.y_size)
            y -= data.y_size;

        return data.input[border_policy_independent::get_idx(x, y, data.x_size)];
    }
};

template <typename Callable>
void apply_border_policy(const std::string& policy_name, Callable func) {
    if (policy_name == "zeros") {
        func.template operator()<zeros_border>();
    }
    else if (policy_name == "wrap_around") {
        func.template operator()<wrap_around_border>();
    }
    else {
        throw std::invalid_argument("Unknown border policy: " + policy_name);
    }
}

} // namespace border_policies

#endif