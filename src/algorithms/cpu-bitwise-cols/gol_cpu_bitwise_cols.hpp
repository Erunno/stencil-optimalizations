#ifndef GOL_CPU_BITWISE_COLS_HPP
#define GOL_CPU_BITWISE_COLS_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise-cols/bit_cols_grid.hpp"
#include "../_shared/bitwise-cols/bitwise_cols_gol_operations.hpp"
#include <cstddef>
#include <memory>

namespace algorithms {

template <std::size_t Bits>
struct BitsConst {};

template <>
struct BitsConst<8> {
    using col_type = std::uint8_t;
};

template <>
struct BitsConst<16> {
    using col_type = std::uint16_t;
};

template <>
struct BitsConst<32> {
    using col_type = std::uint32_t;
};

template <>
struct BitsConst<64> {
    using col_type = std::uint64_t;
};

template <std::size_t Bits>
class GoLCpuBitwiseCols : public infrastructure::Algorithm<2, char> {
  public:
    using col_type = typename BitsConst<Bits>::col_type;
    using BitGrid = algorithms::BitColsGrid<col_type>;
    using DataGrid = infrastructure::Grid<2, char>;
    using size_type = BitGrid::size_type;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        original_x_size = data.size_in<0>();
        original_y_size = data.size_in<1>();
    }

    void initialize_data_structures() override {
        intermediate_bit_grid = std::make_unique<BitGrid>(original_x_size, original_y_size);
    }

    void run(size_type iterations) override {
        auto x_size = bit_grid->x_size();
        auto y_size = bit_grid->y_size();

        auto source = bit_grid.get();
        auto target = intermediate_bit_grid.get();

        std::cout << "Initial state" << std::endl;
        std::cout << source->debug_print(20) << std::endl;

        for (size_type i = 0; i < iterations; ++i) {
            for (size_type y = 0; y < y_size; ++y) {
                for (size_type x = 0; x < x_size; ++x) {
                    // clang-format off

                    col_type lt, ct, rt;
                    col_type lc, cc, rc; 
                    col_type lb, cb, rb;

                    load(source, x, y,
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb);

                    col_type new_center = BitwiseColsOps<col_type>::compute_center_col(
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb
                    );

                    target->set_bit_col(x, y, new_center);

                    // clang-format on
                }
            }
            std::cout << "Iteration: " << i << std::endl;
            std::cout << target->debug_print(20) << std::endl;

            std::swap(source, target);
        }
        final_bit_grid = source;
    }

    void finalize_data_structures() override {
        _result = DataGrid(original_x_size, original_y_size);

        for (size_type y = 0; y < original_y_size; ++y) {
            for (size_type x = 0; x < original_x_size; ++x) {
                auto col = final_bit_grid->get_bit_col(x, y / BitGrid::BITS_IN_COL);
                auto bit = y;

                _result[x][y] = ((col >> bit) & 1) ? 1 : 0;
            }
        }
    }

    DataGrid fetch_result() override {
        return std::move(_result);
    }

  private:
    // clang-format off
    void load(const BitGrid* grid, size_type x, size_type y,
        col_type& lt, col_type& ct, col_type& rt,
        col_type& lc, col_type& cc, col_type& rc,
        col_type& lb, col_type& cb, col_type& rb) {

        load_one(grid, lt, x - 1, y - 1);
        load_one(grid, ct, x,     y - 1);
        load_one(grid, rt, x + 1, y - 1);
        
        load_one(grid, lc, x - 1, y    );
        load_one(grid, cc, x,     y    );
        load_one(grid, rc, x + 1, y    );
        
        load_one(grid, lb, x - 1, y + 1);
        load_one(grid, cb, x,     y + 1);
        load_one(grid, rb, x + 1, y + 1);
    }
    // clang-format on

    void load_one(const BitGrid* grid, col_type& col, size_type x, size_type y) {
        auto x_size = grid->x_size();
        auto y_size = grid->y_size();

        if (x < 0 || x >= x_size || y < 0 || y >= y_size) {
            col = 0;
        }
        else {
            col = grid->get_bit_col(x, y);
        }
    }

    size_type original_x_size;
    size_type original_y_size;

    DataGrid _result;

    std::unique_ptr<BitGrid> bit_grid;
    std::unique_ptr<BitGrid> intermediate_bit_grid;

    BitGrid* final_bit_grid;
};

} // namespace algorithms

#endif // GOL_CPU_BITWISE_COLS_HPP