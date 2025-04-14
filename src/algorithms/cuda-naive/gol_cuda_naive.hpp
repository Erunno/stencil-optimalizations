#ifndef GOL_CUDA_NAIVE_HPP
#define GOL_CUDA_NAIVE_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "../_shared/cuda-helpers/border_policies.cuh"
#include "models.hpp"
#include <cstddef>
#include <iostream>

namespace algorithms {

template <typename grid_cell_t>
class GoLCudaNaive : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
    GoLCudaNaive() = default;

    using size_type = std::size_t;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;

    void set_and_format_input_data(const DataGrid& data) override {
        grid = data;
    }

    void initialize_data_structures() override {
        cuda_data.x_size = grid.template size_in<0>();
        cuda_data.y_size = grid.template size_in<1>();

        auto size = grid.size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(grid_cell_t)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(grid_cell_t)));

        CUCH(cudaMemcpy(cuda_data.input, grid.data(), size * sizeof(grid_cell_t), cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        border_policies::apply_border_policy(this->params.border_mode,
            [this, iterations]<typename policy>() {
                this->run_kernel<policy>(iterations);
            }
        );
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = grid.data();

        CUCH(cudaMemcpy(data, cuda_data.output, grid.size() * sizeof(grid_cell_t), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return std::move(grid);
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    DataGrid grid;
    NaiveGridOnCuda<grid_cell_t> cuda_data;

    template <typename BorderPolicy>
    void run_kernel(size_type iterations);

    void run_kernel(size_type iterations);

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP