#include "pretty_print.hpp"
#include <cstddef>
#include "../infrastructure/colors.hpp"

std::string debug_utils::pretty(const infrastructure::Grid<2, char>& grid) {
    std::string result;

    for (std::size_t y = 0; y < grid.size_in<1>(); ++y) {
        for (std::size_t x = 0; x < grid.size_in<0>(); ++x) {
            std::string cell = "[";
            cell += std::to_string(static_cast<int>(grid[x][y]));
            cell += ']';

            if (grid[x][y] != 0) {
                result += c::grid_print_one() + cell + c::reset_color();
            }
            else {
                result += c::grid_print_zero() + cell + c::reset_color();
            }
        }
        result += "\n";
    }

    return result;
}

std::string debug_utils::pretty(const infrastructure::Grid<3, char>& grid) {
    std::string result;
    auto dimX = grid.size_in<0>();
    auto dimY = grid.size_in<1>();
    auto dimZ = grid.size_in<2>();

    for (std::size_t z = 0; z < dimZ; z++) {
        result += "Layer Z=" + std::to_string(z) + "\n";
        for (std::size_t x = 0; x < dimX; ++x) {
            for (std::size_t y = 0; y < dimY; ++y) {
                result += (grid[x][y][z] == 1 ? "O" : ".");
            }
            result += "\n";
        }
        result += "\n";
    }

    return result;
}
