#ifndef ALGORITHMS_CUDA_LOCAL_ONE_CELL_HPP
#define ALGORITHMS_CUDA_LOCAL_ONE_CELL_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/cuda-helpers/cuch.hpp"
#include "./models.hpp"
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace algorithms::cuda_local_one_cell {


constexpr bool RECORD_CHANGED_TILES = true;
constexpr bool DO_NOT_RECORD_CHANGED_TILES = false;

using StreamingDir = infrastructure::StreamingDirection;

template <typename grid_cell_t, std::size_t Bits, typename state_store_type, typename bit_grid_mode>
class GoLCudaLocalOneCellImpl : public infrastructure::Algorithm<2, grid_cell_t> {

  public:
    GoLCudaLocalOneCellImpl() = default;

    using size_type = std::size_t;
    using word_type = typename BitsConst<Bits>::word_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = GeneralBitGrid<word_type, bit_grid_mode>;
    using BitGrid_ptr = std::unique_ptr<BitGrid>;

    constexpr static std::size_t STATE_STORE_BITS = sizeof(state_store_type) * 8;
    constexpr static std::size_t WARP_SIZE = 32;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);
    }

    void initialize_data_structures() override {
        cuda_data.x_size = bit_grid->x_size();
        cuda_data.y_size = bit_grid->y_size();

        auto size = bit_grid->size();
        
        CUCH(cudaMalloc(&cuda_data.input, size * sizeof(word_type)));
        CUCH(cudaMalloc(&cuda_data.output, size * sizeof(word_type)));

        CUCH(cudaMalloc(&cuda_data.change_state_store.last, state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMalloc(&cuda_data.change_state_store.current, state_store_word_count() * sizeof(state_store_type)));

        CUCH(cudaMemcpy(cuda_data.input, bit_grid->data(), size * sizeof(word_type), cudaMemcpyHostToDevice));

        reset_changed_stores();
    }

    void run(size_type iterations) override {
        if (this->params.collect_touched_tiles_stats) {
            init_changed_tiles_stats(iterations);

            run_kernel<RECORD_CHANGED_TILES>(iterations);

            print_changed_tiles_stats();
        }
        else {
            run_kernel<DO_NOT_RECORD_CHANGED_TILES>(iterations);
        }
    }

    void finalize_data_structures() override {
        CUCH(cudaDeviceSynchronize());

        auto data = bit_grid->data();

        CUCH(cudaMemcpy(data, cuda_data.output, bit_grid->size() * sizeof(word_type), cudaMemcpyDeviceToHost));

        CUCH(cudaFree(cuda_data.input));
        CUCH(cudaFree(cuda_data.output));
        CUCH(cudaFree(cuda_data.change_state_store.last));
        CUCH(cudaFree(cuda_data.change_state_store.current));
    }

    DataGrid fetch_result() override {
        return bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    BitGrid_ptr bit_grid;
    BitGridWithChangeInfo<word_type, state_store_type> cuda_data;

    std::size_t _performed_iterations;

    std::vector<size_type> computed_tiles;
    std::vector<size_type> copied_tiles;
    std::vector<size_type> untouched_tiles;

    template <bool RECORD_CHANGED_TILES>
    void run_kernel(size_type iterations);
    
    std::size_t get_thread_block_count() {
        auto computed_elems_in_block = this->params.thread_block_size; 
        return bit_grid->size() / computed_elems_in_block;
    }

    std::size_t state_store_word_count() {
        return get_thread_block_count();
    }

    void reset_changed_stores() {
        state_store_type zero = 0;

        CUCH(cudaMemset(cuda_data.change_state_store.last, ~zero, state_store_word_count() * sizeof(state_store_type)));
        CUCH(cudaMemset(cuda_data.change_state_store.current, ~zero, state_store_word_count() * sizeof(state_store_type)));
    }

    void rotate_state_stores() {
        std::swap(cuda_data.change_state_store.last, cuda_data.change_state_store.current);
    }

    // DEBUG FUNCTIONS

    void init_changed_tiles_stats(size_type iterations) {
        computed_tiles.resize(iterations, 0);
        copied_tiles.resize(iterations, 0);
        untouched_tiles.resize(iterations, 0);
    }

    void print_changed_tiles_stats() {
        std::cout << "Changed_tiles: ";
        for (std::size_t i = 0; i < _performed_iterations; i++) {
            std::cout << computed_tiles[i] << ";";
        }
        std::cout << std::endl;

        std::cout << "Copied_tiles: ";
        for (std::size_t i = 0; i < _performed_iterations; i++) {
            std::cout << copied_tiles[i] << ";";
        }
        std::cout << std::endl;

        std::cout << "Untouched_tiles: ";
        for (std::size_t i = 0; i < _performed_iterations; i++) {
            std::cout << untouched_tiles[i] << ";";
        }
        std::cout << std::endl;
    }

    void check_state_stores() {
        std::vector<state_store_type> last(state_store_word_count(), 0);
        std::vector<state_store_type> current(state_store_word_count(), 0);

        CUCH(cudaMemcpy(last.data(), cuda_data.change_state_store.last, state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));
        CUCH(cudaMemcpy(current.data(), cuda_data.change_state_store.current, state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));

        std::size_t changed_tiles_in_last = 0;
        std::size_t changed_tiles_in_current = 0;

        for (std::size_t i = 0; i < state_store_word_count(); i++) {
            changed_tiles_in_last += __builtin_popcountll(last[i]);
            changed_tiles_in_current += __builtin_popcountll(current[i]);
        }

        std::cout << "Changed tiles in last: " << changed_tiles_in_last << std::endl;
        std::cout << "Changed tiles in current: " << changed_tiles_in_current << std::endl;

        print_state_store(current.data());

        std::cout << std::endl;
    }

    void print_state_store(state_store_type* store) {
        auto x_tiles = bit_grid->x_size() / 32;
        auto y_tiles = bit_grid->y_size() / 1;

        dim3 block_dims = {
            static_cast<unsigned int>(1),
            static_cast<unsigned int>(this->params.thread_block_size / 32)
        };

        auto x_block_count = x_tiles / block_dims.x;
        auto y_block_count = y_tiles / block_dims.y;

        std::vector<char> output = get_tile_grid(store);

        for (std::size_t y = 0; y < y_tiles; y++) {
            for (std::size_t x = 0; x < x_tiles; x++) {
                std::cout << output[y * x_tiles + x];
            }
            std::cout << std::endl;
        }
    }

    void record_changed_tiles(size_type iter) {
        auto stats = count_computed_tiles_in_next_step();

        computed_tiles[iter] = stats.computed_tiles;
        copied_tiles[iter] = stats.copied_tiles;
        untouched_tiles[iter] = stats.untouched_tiles;
    }

    struct compute_stats {
        size_type computed_tiles;
        size_type copied_tiles;
        size_type untouched_tiles;
    };

    compute_stats count_computed_tiles_in_next_step() {
        std::vector<state_store_type> last(state_store_word_count(), 0);
        std::vector<state_store_type> before_last(state_store_word_count(), 0);
        CUCH(cudaMemcpy(last.data(), cuda_data.change_state_store.last, state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));
        CUCH(cudaMemcpy(before_last.data(), cuda_data.change_state_store.current, state_store_word_count() * sizeof(state_store_type), cudaMemcpyDeviceToHost));

        auto x_tiles = bit_grid->x_size() / 32;
        auto y_tiles = bit_grid->y_size() / 1;

        auto changed_grid = get_tile_grid(last.data());
        auto changed_grid_before_last = get_tile_grid(before_last.data());

        std::size_t computed_tiles = 0;
        std::size_t copied = 0;

        for (std::size_t y = 0; y < y_tiles; y++) {
            for (std::size_t x = 0; x < x_tiles; x++) {
                
                bool is_neighborhood_modified = false;

                for(int dy = -1; dy <= 1; dy++) {
                    for(int dx = -1; dx <= 1; dx++) {

                        auto nx = x + dx;
                        auto ny = y + dy;

                        if (nx >= x_tiles || ny >= y_tiles) {
                            continue;
                        }

                        if (changed_grid[ny * x_tiles + nx] == 'X') {
                            is_neighborhood_modified = true;
                            break;
                        }
                    }
                    if (is_neighborhood_modified) {
                        break;
                    }
                }

                if (is_neighborhood_modified) {
                        computed_tiles++;
                }
                else if (changed_grid_before_last[y * x_tiles + x] == 'X') {
                    copied++;
                }
            }
        }

        return {
            .computed_tiles = computed_tiles,
            .copied_tiles = copied,
            .untouched_tiles = x_tiles * y_tiles - computed_tiles - copied
        };
    }

    std::vector<char> get_tile_grid(state_store_type* store) {
        auto x_tiles = bit_grid->x_size() / 32;
        auto y_tiles = bit_grid->y_size() / 1;

        dim3 block_dims = {
            static_cast<unsigned int>(1),
            static_cast<unsigned int>(this->params.thread_block_size / 32)
        };

        auto x_block_count = x_tiles / block_dims.x;
        auto y_block_count = y_tiles / block_dims.y;

        std::vector<char> output(x_tiles * y_tiles, 0);

        for (std::size_t y_block = 0; y_block < y_block_count; y_block++) {
            for (std::size_t x_block = 0; x_block < x_block_count; x_block++) {
                state_store_type block_state = store[y_block * x_block_count + x_block];

                auto x_base_out = x_block * block_dims.x;
                auto y_base_out = y_block * block_dims.y;

                for (std::size_t y = 0; y < block_dims.y; y++) {
                    for (std::size_t x = 0; x < block_dims.x; x++) {
                        auto bit_idx = y * block_dims.x + x;
                        auto output_idx = (y_base_out + y) * x_tiles + (x_base_out + x);

                        auto tile_changed = (block_state >> bit_idx) & 1;

                        output[output_idx] = tile_changed ? 'X' : '.';
                    }
                }
            }
        }

        return output;
    } 
};


template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
class GoLCudaLocalOneCell : public infrastructure::Algorithm<2, grid_cell_t> {
  public:
    
    void set_and_format_input_data(const infrastructure::Grid<2, grid_cell_t>& data) override {
        fetch_implementation()->set_and_format_input_data(data);
    }

    void initialize_data_structures() override {
        fetch_implementation()->initialize_data_structures();
    }

    void run(std::size_t iterations) override {
        fetch_implementation()->run(iterations);
    }

    void finalize_data_structures() override {
        fetch_implementation()->finalize_data_structures();
    }

    infrastructure::Grid<2, grid_cell_t> fetch_result() override {
        return fetch_implementation()->fetch_result();
    }

    std::size_t actually_performed_iterations() const override {
        return fetch_implementation()->actually_performed_iterations();
    }

    void set_params(const infrastructure::ExperimentParams& params) override {
        this->params = params;
        fetch_implementation()->set_params(params);
    }

  private:
    GoLCudaLocalOneCellImpl<grid_cell_t, Bits, std::uint32_t, bit_grid_mode> implementation_32_state;    
    GoLCudaLocalOneCellImpl<grid_cell_t, Bits, std::uint64_t, bit_grid_mode> implementation_64_state;    

    infrastructure::Algorithm<2, grid_cell_t>* fetch_implementation() {
        if (this->params.state_bits_count == 32) {
            return &implementation_32_state;
        } else if (this->params.state_bits_count == 64) {
            return &implementation_64_state;
        } else {
            throw std::runtime_error("Invalid state bits count");
        }
    }

    const infrastructure::Algorithm<2, grid_cell_t>* fetch_implementation() const {
        if (this->params.state_bits_count == 32) {
            return &implementation_32_state;
        } else if (this->params.state_bits_count == 64) {
            return &implementation_64_state;
        } else {
            throw std::runtime_error("Invalid state bits count");
        }
    }
};

}

#endif // ALGORITHMS_CUDA_LOCAL_ONE_CELL_HPP