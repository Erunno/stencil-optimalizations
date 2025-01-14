#ifndef GOL_CUDA_NAIVE_BITWISE_HPP
#define GOL_CUDA_NAIVE_BITWISE_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <cstddef>
#include <memory>

namespace algorithms {

template <std::size_t Bits>
class GoLCudaNaiveBitwise : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaiveBitwise() = default;

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, char>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        const auto size = bit_grid->size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(col_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(col_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(col_type), cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        run_kernel(iterations);
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        const auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(col_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return bit_grid->to_grid();
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridOnCuda<col_type> cuda_data;

    void run_kernel(size_type iterations);
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP