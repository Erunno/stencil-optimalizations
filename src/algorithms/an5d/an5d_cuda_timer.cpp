#include "an5d_cuda_timer.hpp"

namespace algorithms {

std::chrono::time_point<std::chrono::high_resolution_clock> An5dCudaTimer::start_time;
std::chrono::time_point<std::chrono::high_resolution_clock> An5dCudaTimer::end_time;

} // namespace algorithms
