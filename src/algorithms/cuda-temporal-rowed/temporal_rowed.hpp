#ifndef GOL_CUDA_TEMPORAL_ROWED_HPP
#define GOL_CUDA_TEMPORAL_ROWED_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace algorithms {

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
class GoLCudaTemporalRowed : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
  GoLCudaTemporalRowed() = default;

    using idx_t = std::int64_t;
    using size_type = std::size_t;
    using row_type = typename BitsConst<Bits>::word_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = GeneralBitGrid<row_type, bit_grid_mode>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(row_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(row_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(row_type), cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        auto y_block_size = this->params.thread_block_size / 32;
        auto temporal_steps = this->params.temporal_steps;

        call_correct_kernel(iterations, y_block_size, temporal_steps);
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(row_type), cudaMemcpyDeviceToHost));

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
    RowedGridOnCudaForTemporal<row_type> cuda_data;

    template <idx_t block_y_size, idx_t temporal_steps, idx_t words_per_thread>
    void run_kernel(size_type iterations);

    std::size_t _performed_iterations;

    void call_correct_kernel(size_type iterations, idx_t block_y_size, idx_t temporal_steps) {
        bool err = false;

        run_kernel<16, 11, 10>(iterations);

        // if (temporal_steps == 1) {
        //     if (block_y_size == 32) {
        //         run_kernel<32, 1>(iterations);
        //     }
        //     else if (block_y_size == 16) {
        //         run_kernel<16, 1>(iterations);
        //     }
        //     else if (block_y_size == 8) {
        //         run_kernel<8, 1>(iterations);
        //     }
        //     else if (block_y_size == 4) {
        //         run_kernel<4, 1>(iterations);
        //     }
        //     else {
        //         err = true;
        //     }
        // }
        // else if (temporal_steps == 2) {
        //     if (block_y_size == 32) {
        //         run_kernel<32, 2>(iterations);
        //     }
        //     else if (block_y_size == 16) {
        //         run_kernel<16, 2>(iterations);
        //     }
        //     else if (block_y_size == 8) {
        //         run_kernel<8, 2>(iterations);
        //     }
        //     else if (block_y_size == 4) {
        //         run_kernel<4, 2>(iterations);
        //     }
        //     else {
        //         err = true;
        //     }
        // }
        // else if (temporal_steps == 4) {
        //     if (block_y_size == 32) {
        //         run_kernel<32, 4>(iterations);
        //     }
        //     else if (block_y_size == 16) {
        //         run_kernel<16, 4>(iterations);
        //     }
        //     else if (block_y_size == 8) {
        //         run_kernel<8, 4>(iterations);
        //     }
        //     else if (block_y_size == 4) {
        //         run_kernel<4, 4>(iterations);
        //     }
        //     else {
        //         err = true;
        //     }
        // }
        // else if (temporal_steps == 8) {
        // if (temporal_steps == 8) {
        //     if (block_y_size == 32) {
        //         run_kernel<32, 8>(iterations);
        //     }
        //     else if (block_y_size == 16) {
        //         run_kernel<16, 8>(iterations);
        //     }
        //     else if (block_y_size == 8) {
        //         run_kernel<8, 8>(iterations);
        //     }
        //     else if (block_y_size == 4) {
        //         run_kernel<4, 8>(iterations);
        //     }
        //     else {
        //         err = true;
        //     }
        // }
        // else {
        //     err = true;
        // }

        if (err) {
            std::stringstream error_msg;
            error_msg << "Invalid block size and temporal steps combination: " 
                      << block_y_size << " (y block size) and " << temporal_steps << " (temporal steps)";

            throw std::runtime_error(error_msg.str());
        }
    }
};

} // namespace algorithms

#endif // GOL_CUDA_NAIVE_HPP