#ifndef GOL_CUDA_NAIVE_BITWISE_HPP
#define GOL_CUDA_NAIVE_BITWISE_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "../_shared/cuda-helpers/border_policies.cuh"
#include "./models.hpp"
#include <cstddef>
#include <memory>

namespace algorithms {

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
class GoLCudaNaiveBitwise : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
    GoLCudaNaiveBitwise() = default;

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::word_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = GeneralBitGrid<col_type, bit_grid_mode>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(col_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(col_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(col_type), cudaMemcpyHostToDevice));
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

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(col_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
    }

    DataGrid fetch_result() override {
        return bit_grid->template to_grid<grid_cell_t>();
    }

    
    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridOnCuda<col_type> cuda_data;

    template <typename border_policy>
    void run_kernel(size_type iterations);

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP