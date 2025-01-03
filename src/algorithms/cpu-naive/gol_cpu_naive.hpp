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
    GoLCpuNaive() : animate(false) {};

    using size_type = std::size_t;
    using DataGrid = infrastructure::Grid<2, char>;

    void print_game_of_live_in_progress() {
        this->animate = true;
    }

    void set_and_format_input_data(const DataGrid& data) override {
        _result = data;
    }

    void initialize_data_structures() override {
        _intermediate = _result;
    }

    void run(size_type iterations) override {
        DataGrid* source = &_result;
        DataGrid* target = &_intermediate;

        auto x_size = _result.size_in<0>();
        auto y_size = _result.size_in<1>();

        if (animate) {
            print(*source, 0);
        }

        for (size_type i = 0; i < iterations; ++i) {
            for (size_type x = 1; x < x_size - 1; ++x) {
                for (size_type y = 1; y < y_size - 1; ++y) {

                    auto alive_neighbours = count_alive_neighbours(*source, x, y);

                    if ((*source)[x][y] == 1) {
                        if (alive_neighbours < 2 || alive_neighbours > 3) {
                            (*target)[x][y] = 0;
                        }
                        else {
                            (*target)[x][y] = 1;
                        }
                    }
                    else {
                        if (alive_neighbours == 3) {
                            (*target)[x][y] = 1;
                        }
                        else {
                            (*target)[x][y] = 0;
                        }
                    }
                }
            }

            if (animate) {
                move_cursor_up_left(*target);
                print(*target, i + 1);
            }

            std::swap(target, source);
        }

        _result = *source;
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return _result;
    }

  private:
    size_type count_alive_neighbours(const DataGrid& grid, size_type x, size_type y) {
        size_type alive_neighbours = 0;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                // Skip the cell itself
                if (i == 0 && j == 0)
                    continue;

                auto x_neighbour = x + i;
                auto y_neighbour = y + j;
                alive_neighbours += grid[x_neighbour][y_neighbour] > 0 ? 1 : 0;
            }
        }

        return alive_neighbours;
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
    bool animate;
};

} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP