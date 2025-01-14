#ifndef CUDA_AN5D_GOL_32_64
#define CUDA_AN5D_GOL_32_64

#ifndef CUSTOM_TYPES_DEFINED
#define CUSTOM_TYPES_DEFINED
typedef unsigned int ui32_t;
typedef unsigned long ui64_t;
#endif

#include "an5d_cuda_timer.hpp"
#define STENCILBENCH
#define SB_START_INSTRUMENTS algorithms::An5dCudaTimer::start();
#define SB_STOP_INSTRUMENTS algorithms::An5dCudaTimer::stop();

#undef CELL_TYPE
#define CELL_TYPE ui32_t
void cuda_gol_32(CELL_TYPE* src, CELL_TYPE* dst, int x_size, int y_size, int iters);

#undef CELL_TYPE
#define CELL_TYPE ui64_t
void cuda_gol_64(CELL_TYPE* src, CELL_TYPE* dst, int x_size, int y_size, int iters);

#endif
