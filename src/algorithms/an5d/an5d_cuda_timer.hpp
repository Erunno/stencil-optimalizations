#ifndef AN5D_CUDA_TIMER_HPP
#define AN5D_CUDA_TIMER_HPP

#include <chrono>
#include <iostream>

namespace algorithms {

class An5dCudaTimer {
  public:
    static void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    static void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    static double elapsed_time_s() {
        return std::chrono::duration<double>(end_time - start_time).count();
    }

  private:
    static std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    static std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
};

} // namespace algorithms

#endif // AN5D_CUDA_TIMER_HPP
