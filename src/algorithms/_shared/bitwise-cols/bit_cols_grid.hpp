#ifndef GOL_BIT_COL_GRID_HPP
#define GOL_BIT_COL_GRID_HPP

#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../../debug_utils/pretty_print.hpp"
#include "../../../infrastructure/grid.hpp"

using namespace debug_utils;

namespace algorithms {

template <typename bit_col_type>
class BitColsGrid {
  public:
    constexpr static std::size_t BITS_IN_COL = sizeof(bit_col_type) * 8;

    using size_type = std::size_t;
    using Grid = infrastructure::Grid<2, char>;

    BitColsGrid(size_type original_x_size, size_t original_y_size)
        : _x_size(original_x_size), _y_size(original_y_size) {
        _x_size = original_x_size;
        _y_size = original_y_size / BITS_IN_COL;

        bit_cols_grid.resize(x_size() * y_size(), 0);
    }

    BitColsGrid(const Grid& grid) {
        assert_dim_has_correct_size(grid);

        _x_size = grid.size_in<0>();
        _y_size = grid.size_in<1>() / BITS_IN_COL;

        bit_cols_grid.resize(x_size() * y_size(), 0);
        fill_grid(grid);
    }

  public:
    bit_col_type get_bit_col(std::size_t x, std::size_t y) const {
        return bit_cols_grid[idx(x, y)];
    }

    void set_bit_col(std::size_t x, std::size_t y, bit_col_type bit_col) {
        bit_cols_grid[idx(x, y)] = bit_col;
    }

    std::string debug_print(std::size_t line_limit = std::numeric_limits<std::size_t>::max()) {
        std::ostringstream result;

        for (std::size_t y = 0; y < y_size(); ++y) {
            for (std::size_t bit = 0; bit < BITS_IN_COL; ++bit) {
                for (std::size_t x = 0; x < x_size(); ++x) {
                    auto col = get_bit_col(x, y);
                    char bit_char = ((col >> bit) & 1) ? '1' : '0';

                    result << color_0_1(bit_char);
                    result << " ";
                }
                result << "\n";

                if (y * BITS_IN_COL + bit + 1 >= line_limit) {
                    return result.str();
                }
            }
            result << "\n";
        }

        return result.str();
    }

    size_type x_size() const {
        return _x_size;
    }

    size_type y_size() const {
        return _y_size;
    }

    size_type size() const {
        return x_size() * y_size();
    }

    size_type original_x_size() const {
        return _x_size;
    }

    size_type original_y_size() const {
        return _y_size * BITS_IN_COL;
    }

    bit_col_type* data() {
        return bit_cols_grid.data();
    }

    std::vector<bit_col_type>* data_vector() {
        return &bit_cols_grid;
    }

    Grid to_grid() const {
        auto _original_x_size = original_x_size();
        auto _original_y_size = original_y_size();

        Grid grid(_original_x_size, _original_y_size);
        auto raw_data = grid.data();

        for (size_type y = 0; y < _original_y_size; y += BITS_IN_COL) {
            for (size_type x = 0; x < _original_x_size; ++x) {
                auto col = get_bit_col(x, y / BITS_IN_COL);

                for (size_type bit = 0; bit < BITS_IN_COL; ++bit) {
                    auto value = (col >> bit) & 1;
                    raw_data[in_grid_idx(x, y + bit)] = value;
                }
            }
        }

        return grid;
    }

  private:
    std::size_t in_grid_idx(std::size_t x, std::size_t y) const {
        return y * _x_size + x;
    }

    void assert_dim_has_correct_size(const Grid& grid) {
        if (grid.size_in<1>() % BITS_IN_COL != 0) {
            throw std::invalid_argument("Grid dimensions must be a multiple of " + std::to_string(BITS_IN_COL));
        }
    }

    void fill_grid(const Grid& grid) {
        auto raw_data = grid.data();

        for (std::size_t y = 0; y < grid.size_in<1>(); y += BITS_IN_COL) {
            for (std::size_t x = 0; x < grid.size_in<0>(); ++x) {
                bit_col_type bit_col = 0;

                for (std::size_t i = 0; i < BITS_IN_COL; ++i) {
                    auto value = static_cast<bit_col_type>(raw_data[in_grid_idx(x, y + i)]);
                    bit_col |= value << i;
                }

                set_bit_col(x, y / BITS_IN_COL, bit_col);
            }
        }
    }

    std::string color_0_1(char ch) {
        if (ch == '0') {
            return "\033[30m" + std::string(1, ch) + "\033[0m";
        }
        else {
            return "\033[31m" + std::string(1, ch) + "\033[0m";
        }
    }

    std::size_t idx(std::size_t x, std::size_t y) const {
        return y * x_size() + x;
    }

    std::vector<bit_col_type> bit_cols_grid;
    size_type _x_size;
    size_type _y_size;
};

} // namespace algorithms

#endif // GOL_BIT_COL_GRID_HPP