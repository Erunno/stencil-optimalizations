#ifndef ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP
#define ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP

#include <bitset>
#include <cstddef>
#include <iostream>
#include <sstream>

#include "../template_helpers/static_for.hpp"

namespace algorithms {

enum class Position {
    TOP = 0,
    BOTTOM = 1,
};

template <Position POSITION, typename col_type>
class MasksByPosition {};

template <typename col_type>
struct BitwiseColsOps {
    constexpr static std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    template <Position POSITION>
    using masks = MasksByPosition<POSITION, col_type>;

    // clang-format off
    static col_type compute_center_col(
        col_type lt, col_type ct, col_type rt, 
        col_type lc, col_type cc, col_type rc,
        col_type lb, col_type cb, col_type rb) {
        
        col_type result = compute_inner_bits(lc, cc, rc);

        result |= compute_side_col<Position::TOP>(
            lc, cc, rc,
            lt, ct, rt);

        result |= compute_side_col<Position::BOTTOM>(
            lc, cc, rc,
            lb, cb, rb);

        return result; 
    }
    // clang-format on

  private:
    static col_type compute_inner_bits(col_type lc, col_type cc, col_type rc) {
        col_type result = 0;

        templates::static_for<1, BITS_IN_COL - 1>::run(
            [lc, cc, rc, &result]<std::size_t N>() { result |= compute_inner_cell<N>(lc, cc, rc); });

        return result;
    }

    template <std::size_t N>
    static col_type compute_inner_cell(col_type lc, col_type cc, col_type rc) {
        col_type result = 0;
        constexpr col_type cell_mask = static_cast<col_type>(0b010) << (N - 1);
        constexpr col_type one = cell_mask;

        const auto cell = cc & cell_mask;

        const auto neighborhood = combine_neighborhoods_into_one_word<N>(lc, cc, rc);
        const auto alive_neighbors = __builtin_popcountll(neighborhood);

        // auto alive_neighbors =
        //     __builtin_popcountll(lc & cell_mask) +
        //     __builtin_popcountll(cc & cell_mask) +
        //     __builtin_popcountll(rc & cell_mask);

        if (cell != 0) {
            if (alive_neighbors < 2 || alive_neighbors > 3) {
                result &= ~one;
            }
            else {
                result |= one;
            }
        }
        else {
            if (alive_neighbors == 3) {
                result |= one;
            }
            else {
                result &= ~one;
            }
        }

        return result;
    }

    template <std::size_t N>
    static col_type combine_neighborhoods_into_one_word(col_type lc, col_type cc, col_type rc) {

        constexpr col_type site_neighborhood_mask = static_cast<col_type>(0b111) << (N - 1);
        constexpr col_type center_neighborhood_mask = static_cast<col_type>(0b101) << (N - 1);
        constexpr col_type NEIGHBORHOOD_WINDOW = 6;

        return offset<6, N - 1, NEIGHBORHOOD_WINDOW>(lc & site_neighborhood_mask) |
               offset<3, N - 1, NEIGHBORHOOD_WINDOW>(cc & center_neighborhood_mask) | (rc & site_neighborhood_mask);
    }

    template <std::size_t N, std::size_t CENTER, std::size_t NEIGHBORHOOD_WINDOW>
    static col_type offset(col_type num) {
        if constexpr (CENTER < NEIGHBORHOOD_WINDOW) {
            return num << N;
        }
        else {
            return num >> N;
        }
    }

    // clang-format off
    template <Position POSITION>
    static col_type compute_side_col(
        col_type cl, col_type cc, col_type cr,
        col_type l_, col_type c_, col_type r_) {

        constexpr col_type SITE_MASK = masks<POSITION>::SITE_MASK;
        constexpr col_type CENTER_MASK = masks<POSITION>::CENTER_MASK;
        constexpr col_type UP_BOTTOM_MASK = masks<POSITION>::UP_BOTTOM_MASK;
        constexpr col_type CELL_MASK = masks<POSITION>::CELL_MASK;

        auto neighborhood = 
            masks<POSITION>::template offset_center_cols<7>(cl & SITE_MASK) | 
            masks<POSITION>::template offset_center_cols<5>(cc & CENTER_MASK) |
            masks<POSITION>::template offset_center_cols<3>(cr & SITE_MASK) |
            masks<POSITION>::template offset_top_bottom_cols<2>(l_ & UP_BOTTOM_MASK) |
            masks<POSITION>::template offset_top_bottom_cols<1>(c_ & UP_BOTTOM_MASK) |
                                                               (r_ & UP_BOTTOM_MASK);
        
        auto cell = cc & CELL_MASK;

        // auto alive_neighbors = 
        //     __builtin_popcountll(cl & SITE_MASK) +
        //     __builtin_popcountll(cc & CENTER_MASK) +
        //     __builtin_popcountll(cr & SITE_MASK) +
        //     __builtin_popcountll(l_ & UP_BOTTOM_MASK) +
        //     __builtin_popcountll(c_ & UP_BOTTOM_MASK) +
        //     __builtin_popcountll(r_ & UP_BOTTOM_MASK);

        auto alive_neighbors = __builtin_popcountll(neighborhood);

        if (cell != 0) {
            if (alive_neighbors < 2 || alive_neighbors > 3) {
                return 0;
            }
            else {
                return CELL_MASK;
            }
        }
        else {
            if (alive_neighbors == 3) {
                return CELL_MASK;
            }
            else {
                return 0;
            }
        }

    }
    // clang-format on
};

template <typename col_type>
class MasksByPosition<Position::TOP, col_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    static constexpr col_type SITE_MASK = 0b11;
    static constexpr col_type CENTER_MASK = 0b10;
    static constexpr col_type UP_BOTTOM_MASK = static_cast<col_type>(1) << (BITS_IN_COL - 1);
    static constexpr col_type CELL_MASK = 0b1;

    template <std::size_t N>
    static col_type offset_center_cols(col_type num) {
        return num << N;
    }

    template <std::size_t N>
    static col_type offset_top_bottom_cols(col_type num) {
        return num >> N;
    }
};

template <typename col_type>
class MasksByPosition<Position::BOTTOM, col_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(col_type) * 8;

    static constexpr col_type SITE_MASK = static_cast<col_type>(0b11) << (BITS_IN_COL - 2);
    static constexpr col_type CENTER_MASK = static_cast<col_type>(0b01) << (BITS_IN_COL - 2);
    static constexpr col_type UP_BOTTOM_MASK = 1;
    static constexpr col_type CELL_MASK = static_cast<col_type>(0b1) << (BITS_IN_COL - 1);

    template <std::size_t N>
    static col_type offset_center_cols(col_type num) {
        return num >> N;
    }

    template <std::size_t N>
    static col_type offset_top_bottom_cols(col_type num) {
        return num << N;
    }
};

} // namespace algorithms

#endif // ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP