#ifndef GOL_CUDA_NAIVE_LOCAL_HPP
#define GOL_CUDA_NAIVE_LOCAL_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_col_types.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>

namespace algorithms::cuda_naive_local {

template <std::size_t Bits, typename state_store_type = std::uint32_t>
class GoLCudaNaiveLocal : public infrastructure::Algorithm<2, char> {

  public:
    GoLCudaNaiveLocal() = default;

    using size_type = std::size_t;
    using col_type = typename BitsConst<Bits>::col_type;
    using DataGrid = infrastructure::Grid<2, char>;
    using BitGrid = BitColsGrid<col_type>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    constexpr static std::size_t STATE_STORE_BITS = sizeof(state_store_type) * 8;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        cuda_data.warp_dims = {.x = 8, .y = 4};
        cuda_data.warp_tile_dims = {.x = 16, .y = 8};
        thread_block_size = 32 * 8;

        assert(warp_size() == 32);
        assert(tiles_per_block() < STATE_STORE_BITS);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();

        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(col_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(col_type)));

        CUCH(
            cudaMalloc(&cuda_data.change_state_store.before_last, state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMalloc(&cuda_data.change_state_store.last, state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMalloc(&cuda_data.change_state_store.current, state_store_word_count() * sizeof(state_store_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(col_type), cudaMemcpyHostToDevice));

        reset_changed_stores();
    }

    void run(size_type iterations) override {
        run_kernel(iterations);
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(col_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
        CUCH(cudaFree(cuda_data.change_state_store.before_last));
        CUCH(cudaFree(cuda_data.change_state_store.last));
        CUCH(cudaFree(cuda_data.change_state_store.current));
    }

    DataGrid fetch_result() override {
        return bit_grid->to_grid();
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridWithChangeInfo<col_type, state_store_type> cuda_data;

    std::size_t thread_block_size;

    void run_kernel(size_type iterations);

    std::size_t warp_tile_size() const {
        return cuda_data.warp_tile_dims.x * cuda_data.warp_tile_dims.y;
    }

    std::size_t warp_size() const {
        return cuda_data.warp_dims.x * cuda_data.warp_dims.y;
    }

    std::size_t get_warp_tiles_count() {
        return bit_grid->size() / warp_tile_size();
    }

    std::size_t get_thread_block_count() {
        auto warps_per_block = thread_block_size / warp_size();
        auto computed_elems_in_block = warps_per_block * warp_tile_size();

        return bit_grid->size() / computed_elems_in_block;
    }

    std::size_t state_store_word_count() {
        return get_thread_block_count();
    }

    void reset_changed_stores() {
        state_store_type zero = 0;

        CUCH(cudaMemset(cuda_data.change_state_store.before_last, ~zero,
                        state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMemset(cuda_data.change_state_store.last, ~zero, state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMemset(cuda_data.change_state_store.current, ~zero,
                        state_store_word_count() * sizeof(state_store_type)));
    }

    void check_state_stores() {
        std::vector<state_store_type> last(state_store_word_count(), 0);
        std::vector<state_store_type> current(state_store_word_count(), 0);

        CUCH(cudaMemcpy(last.data(), cuda_data.change_state_store.last,
                        state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));
        CUCH(cudaMemcpy(current.data(), cuda_data.change_state_store.current,
                        state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));

        std::size_t changed_tiles_in_last = 0;
        std::size_t changed_tiles_in_current = 0;

        for (std::size_t i = 0; i < state_store_word_count(); i++) {
            changed_tiles_in_last += __builtin_popcountll(last[i]);
            changed_tiles_in_current += __builtin_popcountll(current[i]);
        }

        print_state_store(current.data());

        std::cout << std::endl;
    }

    std::size_t tiles_per_block() {
        return thread_block_size / warp_size();
    }

    void rotate_state_stores() {
        std::swap(cuda_data.change_state_store.before_last, cuda_data.change_state_store.last);
        std::swap(cuda_data.change_state_store.last, cuda_data.change_state_store.current);
    }

    void print_state_store(state_store_type* store) {
        const auto x_tiles = bit_grid->x_size() / cuda_data.warp_tile_dims.x;
        const auto y_tiles = bit_grid->y_size() / cuda_data.warp_tile_dims.y;

        constexpr std::size_t MAX_WIDTH = 32 * 4;
        const auto shrink_factor = x_tiles / MAX_WIDTH;

        const auto used_bits_in_word = tiles_per_block();

        for (std::size_t y = 0; y < y_tiles; y += shrink_factor) {
            for (std::size_t x = 0; x < x_tiles; x += shrink_factor) {
                bool changed = false;

                for (std::size_t i_y = y; i_y < y + shrink_factor; i_y++) {
                    for (std::size_t i_x = x; i_x < x + shrink_factor; i_x++) {
                        const auto idx = i_y * x_tiles + i_x;

                        const auto word_idx = idx / used_bits_in_word;
                        const auto bit_idx = idx % used_bits_in_word;

                        const auto word = store[word_idx];
                        const auto bit = (word >> bit_idx) & 1;

                        if (bit) {
                            changed = true;
                            break;
                        }
                    }
                }

                if (changed) {
                    std::cout << "X";
                }
                else {
                    std::cout << ".";
                }
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }
};

} // namespace algorithms::cuda_naive_local

#endif // GOL_CUDA_NAIVE_HPP
