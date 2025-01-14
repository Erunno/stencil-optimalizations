#ifndef TIMER_HPP
#define TIMER_HPP

#include "../algorithms/_shared/cuda-helpers/cuch.hpp"
#include <chrono>
#include <cuda_runtime.h>

namespace infrastructure {

class Timer {

  public:
    template <typename Func>
    static double measure(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();

        func();
        CUCH(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration<double>(end - start).count();
    }
};

} // namespace infrastructure

#endif // TIMER_HPP
