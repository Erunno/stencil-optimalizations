#ifndef DIFF_GRID_HPP
#define DIFF_GRID_HPP

#include <cstddef>
#include <sstream>
#include <string>

#include "../infrastructure/grid.hpp"
#include "./pretty_print.hpp"

namespace debug_utils {

template <int Dims, typename ElementType>
std::string diff(const infrastructure::Grid<Dims, ElementType>& original,
                        const infrastructure::Grid<Dims, ElementType>& other) {

    std::ostringstream diff_str;

    auto original_data = original.data();
    auto other_data = other.data();

    for (size_t i = 0; i < original.size(); i++) {
        if (original_data[i] != other_data[i]) {
            auto coords = original.idx_to_coordinates(i);

            diff_str << "at: " << i << " ~ " << pretty(coords) << ": " << std::to_string(original_data[i])
                     << " != \033[31m" << std::to_string(other_data[i]) << "\033[0m" << std::endl;
        }
    }
    return diff_str.str();
}

template <typename ElementType>
std::string diff(const infrastructure::Grid<2, ElementType>& original,
                 const infrastructure::Grid<2, ElementType>& other) {

    std::ostringstream diff_str;

    auto x_size = original.size_in(0);
    auto y_size = original.size_in(1);

    const std::size_t printed_tiles_size = 8;

    for (std::size_t y = 0; y < y_size; y++) {
        for (std::size_t x = 0; x < x_size; x++) {

            if ((x % printed_tiles_size == 0) && (x != 0)) {
                diff_str << " ";
            }

            if (x != 0) {
                diff_str << " ";
            }

            if (original[x][y] != other[x][y]) {
                // diff_str << "\033[31m" << std::to_string(other[x][y]) << "\033[34m" << std::to_string(original[x][y])
                // << "\033[0m";
                diff_str << "\033[31m" << std::to_string(other[x][y]) << "\033[0m";
            }
            else {
                auto color = original[x][y] == 0 ? "\033[30m" : "\033[33m";
                diff_str << color << std::to_string(original[x][y]) << "\033[0m";
            }
        }

        if ((y + 1) % printed_tiles_size == 0) {
            diff_str << "\n";
        }

        diff_str << "\n";
    }

    return diff_str.str();
}

template <int Dims, typename ElementType>
std::string diff_on_big_grid(const infrastructure::Grid<Dims, ElementType>& original,
                             const infrastructure::Grid<Dims, ElementType>& other) {
    (void)original;
    (void)other;
    return "Diff on big grid not implemented";
}

template <typename ElementType>
std::string diff_on_big_grid(const infrastructure::Grid<2, ElementType>& original,
                             const infrastructure::Grid<2, ElementType>& other) {
    std::ostringstream diff_str;

    auto x_size = original.size_in(0);
    auto y_size = original.size_in(1);
    
    // Number of rows and columns to display in the contracted view
    const std::size_t display_rows = 32;
    const std::size_t display_cols = 32;
    
    // Calculate tile dimensions
    auto tile_width = (x_size + display_cols - 1) / display_cols;  // Ceiling division
    auto tile_height = (y_size + display_rows - 1) / display_rows;  // Ceiling division
    
    diff_str << "Contracted grid view (" << x_size << "x" << y_size << " → " 
             << display_cols << "x" << display_rows << "), tile size: " 
             << tile_width << "x" << tile_height << "\n\n";
    
    // For each tile in the contracted grid
    for (std::size_t tile_y = 0; tile_y < display_rows && tile_y * tile_height < y_size; tile_y++) {
        for (std::size_t tile_x = 0; tile_x < display_cols && tile_x * tile_width < x_size; tile_x++) {
            // Set spacing between tiles
            if (tile_x != 0) {
                diff_str << " ";
            }
            
            if ((tile_x % 8 == 0) && (tile_x != 0)) {
                diff_str << " ";
            }

            // Calculate tile boundaries
            std::size_t start_x = tile_x * tile_width;
            std::size_t end_x = std::min(start_x + tile_width, x_size);
            std::size_t start_y = tile_y * tile_height;
            std::size_t end_y = std::min(start_y + tile_height, y_size);

            // Check if there's any difference in this tile
            bool different = false;
            for (std::size_t y = start_y; y < end_y && !different; y++) {
                for (std::size_t x = start_x; x < end_x && !different; x++) {
                    if (original[x][y] != other[x][y]) {
                        different = true;
                    }
                }
            }

            // Print tile representation
            if (different) {
                diff_str << "\033[31mX\033[0m";  // Red X for different tiles
            } else {
                diff_str << "\033[32m·\033[0m";  // Green dot for matching tiles
            }
        }

        // Add spacing between rows
        if ((tile_y + 1) % 8 == 0 && tile_y + 1 < display_rows) {
            diff_str << "\n";
        }
        diff_str << "\n";
    }

    return diff_str.str();
}

}; // namespace debug_utils

#endif // DIFF_GRID_HPP
