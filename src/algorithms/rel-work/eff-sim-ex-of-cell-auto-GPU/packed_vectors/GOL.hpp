#ifndef rel_work_GOL_vectors_packed_HPP
#define rel_work_GOL_vectors_packed_HPP

#include "../../../../infrastructure/algorithm.hpp"
#include "../_shared/grid_transformations.hpp"
#include <cstddef>
#include "../../../_shared/cuda-helpers/block_to_2dim.hpp"
#include <iostream>

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

struct _32_bit_policy_vectors {
    static constexpr std::size_t ELEMENTS_PER_CELL = 4;
    
    static std::size_t ROW_SIZE(std::size_t grid_size) {
        return grid_size / ELEMENTS_PER_CELL;
    }

    using CELL_TYPE = uint32_t;

    static constexpr std::size_t sizeof_CELL_TYPE = sizeof(CELL_TYPE);
};

struct _64_bit_policy_vectors {
    static constexpr std::size_t ELEMENTS_PER_CELL = 8;

    static std::size_t ROW_SIZE(std::size_t grid_size) {
        return grid_size / ELEMENTS_PER_CELL;
    }

    using CELL_TYPE = uint64_t;

    static constexpr std::size_t sizeof_CELL_TYPE = sizeof(CELL_TYPE);
};

template <typename grid_cell_t, typename policy>
class GOL_Packed_vectors : public infrastructure::Algorithm<2, grid_cell_t> {
  public:

    using size_type = std::size_t;

    template <typename cell_t>
    using DataGrid = infrastructure::Grid<2, cell_t>;

    using CELL_TYPE = typename policy::CELL_TYPE;
    static constexpr std::size_t ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;
    

    void set_and_format_input_data(const DataGrid<grid_cell_t>& data) override {
        input_output_data_grid = GridTransformer::transform_grid_with_halo<char, grid_cell_t, policy::sizeof_CELL_TYPE>(data);
    }

    void initialize_data_structures() override {
        auto bytes = sizeof(CELL_TYPE)*(GRID_SIZE+2)*(ROW_SIZE+2);

        CUCH(cudaMalloc(&grid, bytes));
        CUCH(cudaMalloc(&new_grid, bytes));

        CUCH(cudaMemcpy(grid, input_output_data_grid.data(), bytes, cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(new_grid, input_output_data_grid.data(), bytes, cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        run_kernel(iterations);
    }

    void finalize_data_structures() override {
        auto bytes = sizeof(CELL_TYPE)*(GRID_SIZE+2)*(ROW_SIZE+2);

        CUCH(cudaMemcpy(input_output_data_grid.data(), new_grid, bytes, cudaMemcpyDeviceToHost));

        CUCH(cudaFree(grid));
        CUCH(cudaFree(new_grid));
    }

    DataGrid<grid_cell_t> fetch_result() override {
        return GridTransformer::transform_grid_remove_halo<grid_cell_t, char, policy::sizeof_CELL_TYPE>(input_output_data_grid);
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

    void set_params(const infrastructure::ExperimentParams& params) override {
        this->params = params;

        if (params.grid_dimensions[0] != params.grid_dimensions[1]) {
            throw std::runtime_error("Only square grids are supported");
        }

        auto block = get_2d_block(this->params.thread_block_size);

        if (block.x != block.y) {
            throw std::runtime_error("Only square thread blocks are supported");
        }

        BLOCK_SIZE = block.x;
        GRID_SIZE = params.grid_dimensions[0];
        ROW_SIZE = policy::ROW_SIZE(GRID_SIZE);
    }

  private:
    CELL_TYPE* grid;
    CELL_TYPE* new_grid;
    std::size_t GRID_SIZE;
    int BLOCK_SIZE;
    std::size_t ROW_SIZE;

    size_type _performed_iterations;

    DataGrid<char> input_output_data_grid;

    void run_kernel(size_type iterations);
};

}

#endif // rel_work_GOL_vectors_packed_HPP