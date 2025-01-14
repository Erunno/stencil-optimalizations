#ifndef DEBUG_UTILS_PRETTY_PRINT_HPP
#define DEBUG_UTILS_PRETTY_PRINT_HPP

#include "../infrastructure/grid.hpp"
#include <string>
#include <vector>

namespace debug_utils {

template <typename T>
std::string pretty(const std::vector<T>& vec) {
    std::string result = "[";
    for (auto& elem : vec) {
        result += std::to_string(elem) + ", ";
    }
    result.pop_back();
    result.pop_back();
    result += "]";
    return result;
}

std::string pretty(const infrastructure::Grid<2, char>& grid);
std::string pretty(const infrastructure::Grid<3, char>& grid);

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
std::string pretty(T& elem) {
    return std::to_string(elem);
}

} // namespace debug_utils

#endif // DEBUG_UTILS_PRETTY_PRINT_HPP
