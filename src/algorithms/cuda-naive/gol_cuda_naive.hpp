#ifndef GOL_CUDA_NAIVE_HPP
#define GOL_CUDA_NAIVE_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "models.hpp"
#include <iostream>

namespace algorithms {

class GoLCudaNaive : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaive() = default;

    using size_type = std::size_t;
    using DataGrid = infrastructure::Grid<2, char>;

    void set_and_format_input_data(const DataGrid& data) override {
        grid = data;
    }

    void initialize_data_structures() override {
        cuda_data.x_size = grid.size_in<0>();
        cuda_data.y_size = grid.size_in<1>();

        const auto size = grid.size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(char)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(char)));

        CUCH(cudaMemcpy(cuda_data.input, grid.data(), size * sizeof(char), cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        run_kernel(iterations);
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        const auto data = grid.data();

        CUCH(cudaMemcpy(data, cuda_data.output, grid.size() * sizeof(char), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return std::move(grid);
    }

  private:
    DataGrid grid;
    NaiveGridOnCuda cuda_data;

    void run_kernel(size_type iterations);
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP