#ifndef GOL_CPU_NAIVE_HPP
#define GOL_CPU_NAIVE_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include <chrono>
#include <iostream>
#include <thread>

namespace algorithms {

class GoLCpuNaive : public infrastructure::Algorithm<2, char> {
  public:
    using size_type = std::size_t;
    using DataGrid = infrastructure::Grid<2, char>;

    void set_and_format_input_data(const DataGrid& data) override {
        _result = data;
    }

    void initialize_data_structures() override {
        _intermediate = _result;
    }

    void run(size_type iterations) override {
        DataGrid* source = &_result;
        DataGrid* target = &_intermediate;

        const auto x_size = _result.size_in<0>();
        const auto y_size = _result.size_in<1>();

        if (this->params.animate_output) {
            print(*source, 0);
        }

        for (size_type i = 0; i < iterations; ++i) {
            for (size_type x = 0; x < x_size; ++x) {
                for (size_type y = 0; y < y_size; ++y) {

                    const auto alive_neighbors = count_alive_neighbors(*source, x, y);

                    if ((*source)[x][y] == 1) {
                        if (alive_neighbors < 2 || alive_neighbors > 3) {
                            (*target)[x][y] = 0;
                        }
                        else {
                            (*target)[x][y] = 1;
                        }
                    }
                    else {
                        if (alive_neighbors == 3) {
                            (*target)[x][y] = 1;
                        }
                        else {
                            (*target)[x][y] = 0;
                        }
                    }
                }
            }

            if (this->params.animate_output) {
                print(*target, i + 1);
            }

            std::swap(target, source);
        }

        _result = *source;
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return std::move(_result);
    }

  private:
    size_type count_alive_neighbors(const DataGrid& grid, size_type x, size_type y) {
        size_type alive_neighbors = 0;

        size_type x_size = grid.size_in<0>();
        size_type y_size = grid.size_in<1>();

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                // Skip the cell itself
                if (i == 0 && j == 0)
                    continue;

                const auto x_neighbor = x + i;
                const auto y_neighbor = y + j;

                constexpr std::size_t zero = 0;

                if (x_neighbor < zero || x_neighbor >= x_size || y_neighbor < zero || y_neighbor >= y_size)
                    continue;

                alive_neighbors += grid[x_neighbor][y_neighbor] > 0 ? 1 : 0;
            }
        }

        return alive_neighbors;
    }

    void move_cursor_up_left(const DataGrid& grid) {
        std::cout << "\033[" << grid.size_in<0>() + 2 << "A";
        std::cout << "\033[" << grid.size_in<1>() << "D";
    }

    void print(const DataGrid& grid, size_type iter) {

        std::cout << "Iteration: " << iter << std::endl;

        std::cout << debug_utils::pretty(grid) << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DataGrid _result;
    DataGrid _intermediate;
};

} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP