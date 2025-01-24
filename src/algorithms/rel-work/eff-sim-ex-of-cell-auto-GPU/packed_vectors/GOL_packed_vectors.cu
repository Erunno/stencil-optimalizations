#include "GOL.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

template <typename policy, typename CELL_TYPE>
__global__ void GOL_packed_vectors (int GRID_SIZE, const CELL_TYPE *__restrict__ grid, CELL_TYPE *__restrict__ newGrid)
{
    constexpr int ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;

    constexpr CELL_TYPE vones = static_cast<CELL_TYPE>(-1) / static_cast<CELL_TYPE>(0xFFU);

    constexpr CELL_TYPE vtwos = 2U * vones;
    constexpr CELL_TYPE vfours = 4U * vones;
    constexpr CELL_TYPE veights = 8U * vones;

    const int ROW_SIZE = GRID_SIZE / ELEMENTS_PER_CELL;

    constexpr unsigned int shift_next = (ELEMENTS_PER_CELL-1)*8;

    // We want id ∈ [1,SIZE]
    const int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    const int id = iy * (ROW_SIZE+2) + ix;

    if (iy<=0 || iy > GRID_SIZE || ix<=0 || ix > ROW_SIZE) {
        return;
    }

    auto&& cell = grid[id];

    auto&& up_cell = grid[id-(ROW_SIZE+2)];
    auto&& down_cell = grid[id+(ROW_SIZE+2)];

    auto&& left_cell = grid[id+1];
    auto&& upleft_cell = grid[id-(ROW_SIZE+1)];
    auto&& downleft_cell = grid[id+(ROW_SIZE+3)];

    auto&& right_cell = grid[id-1];
    auto&& upright_cell = grid[id-(ROW_SIZE+3)];
    auto&& downright_cell = grid[id+(ROW_SIZE+1)];

    const CELL_TYPE numNeighbors =
        (up_cell >> 8U) + up_cell + (up_cell << 8U) +
        (down_cell >> 8U) + down_cell + (down_cell << 8U) +
        (left_cell << shift_next) + (upleft_cell << shift_next) + (downleft_cell << shift_next) +
        (right_cell >> shift_next) + (upright_cell >> shift_next) + (downright_cell >> shift_next) +
        (cell >> 8U) + (cell << 8U);

    // const auto alive_rule = __vcmpeq4(numNeighbors, vtwos) & cell;
    // const auto general_rule = __vcmpeq4(numNeighbors, threes) & vones;
    // newGrid[id] = alive_rule | general_rule;

    const auto bit2 = ((numNeighbors & vtwos) >> 1U) & ~((numNeighbors & vfours) >> 2U) & ~((numNeighbors & veights) >> 3U);
    const auto bit1 = (numNeighbors & vones) | cell;
    newGrid[id] = bit1 & bit2;
}

} // namespace

template <typename grid_cell_t, typename policy>
void GOL_Packed_vectors<grid_cell_t, policy>::run_kernel(size_type iterations) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGridx = (int)ceil(ROW_SIZE/(float)BLOCK_SIZE);
    int linGridy = (int)ceil(GRID_SIZE/(float)BLOCK_SIZE);
    dim3 gridSize(linGridx,linGridy,1);
 
    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }
        
        if (i != 0) {
            std::swap(grid, new_grid);
        }

        GOL_packed_vectors<policy><<<gridSize, blockSize>>>(GRID_SIZE, grid, new_grid);
        CUCH(cudaPeekAtLastError());
    }
}


} // namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy_vectors>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::INT, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy_vectors>;

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_64_bit_policy_vectors>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_vectors<common::INT, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_64_bit_policy_vectors>;
