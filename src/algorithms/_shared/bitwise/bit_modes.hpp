#ifndef ALGORITHMS_BITWISE_BIT_MODES_HPP
#define ALGORITHMS_BITWISE_BIT_MODES_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace algorithms {

template <typename bit_type>
struct BitColumns {
    constexpr static std::size_t X_BITS = 1;
    constexpr static std::size_t Y_BITS = sizeof(bit_type) * 8;

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        (void)x;

        bit_type one = 1;
        auto y_bit_pos = y % Y_BITS;

        return one << y_bit_pos;
    }

    constexpr static bit_type first_mask = 1;
    static bit_type move_next_mask(bit_type mask) {
        return mask << 1;
    }

    static std::string name() {
        return "Columns " + std::to_string(Y_BITS) + " bits policy";
    }
};

template <typename bit_type>
struct BitTile {};

struct BitTileCommon {
    template <std::size_t x_size, std::size_t y_size, typename bit_type>
    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        x = x % x_size;
        y = y % y_size;

        auto shift = y * x_size + x;
        
        constexpr bit_type one = static_cast<bit_type>(1) << (sizeof(bit_type) * 8 - 1);
        return one >> shift;
    }

    template <typename bit_type>
    static bit_type move_next_mask(bit_type mask) {
        return mask >> 1;
    }

    template <typename bit_type>
    constexpr static bit_type first_mask = static_cast<bit_type>(1) << (sizeof(bit_type) * 8 - 1);
};

template <>
struct BitTile<std::uint16_t> {
    using bit_type = std::uint16_t;

    constexpr static std::size_t X_BITS = 4;
    constexpr static std::size_t Y_BITS = 4;
    

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }
    
    static std::string name() {
        return "Tiles " + std::to_string(X_BITS) + "x" + std::to_string(Y_BITS) + " bits policy";
    }
};

template <>
struct BitTile<std::uint32_t> {
    using bit_type = std::uint32_t;

    constexpr static std::size_t X_BITS = 8;
    constexpr static std::size_t Y_BITS = 4;

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }

    
    static std::string name() {
        return "Tiles " + std::to_string(X_BITS) + "x" + std::to_string(Y_BITS) + " bits policy";
    }
};

template <>
struct BitTile<std::uint64_t> {
    using bit_type = std::uint64_t;

    constexpr static std::size_t X_BITS = 8;
    constexpr static std::size_t Y_BITS = 8;

    constexpr static bit_type first_mask = BitTileCommon::first_mask<bit_type>; 

    static bit_type move_next_mask(bit_type mask) {
        return BitTileCommon::move_next_mask(mask);
    }

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        return BitTileCommon::get_bit_mask_for<X_BITS, Y_BITS, bit_type>(x, y);
    }

    
    static std::string name() {
        return "Tiles " + std::to_string(X_BITS) + "x" + std::to_string(Y_BITS) + " bits policy";
    }
};

template <typename bit_type, int bits_per_cell>
struct GeneralPackedRows {
    constexpr static std::size_t BITS = sizeof(bit_type) * 8;
    constexpr static std::size_t CELLS_PER_WORD = BITS / bits_per_cell;
    constexpr static std::size_t BITS_PER_CELL = bits_per_cell;

    constexpr static std::size_t X_BITS = CELLS_PER_WORD;
    constexpr static std::size_t Y_BITS = 1;

    static bit_type get_bit_mask_for(std::size_t x, std::size_t y) {
        (void)y;

        bit_type one = 1;
        auto x_bit_pos = x % X_BITS;

        return one << (x_bit_pos * BITS_PER_CELL);
    }

    constexpr static bit_type first_mask = 1;
    static bit_type move_next_mask(bit_type mask) {
        return mask << BITS_PER_CELL;
    }

    static std::string name() {
        return "Packed rows " + std::to_string(BITS) + " bits policy, " 
            + std::to_string(BITS_PER_CELL) + " bits per cell";
    }
};

template <typename bit_type>
using WastefulRows = GeneralPackedRows<bit_type, 4>;

template <typename bit_type>
using HalfPackedRows = GeneralPackedRows<bit_type, 2>;

template <typename bit_type>
using FullyPackedRows = GeneralPackedRows<bit_type, 1>;

struct BitColumnsMode {
    template <typename bit_type>
    using policy = BitColumns<bit_type>;

    static std::string name() {
        return "ModeColumns";
    }
};

struct BitTileMode {
    template <typename bit_type>
    using policy = BitTile<bit_type>;

    static std::string name() {
        return "ModeTiles";
    }
};

struct BitWastefulRowsMode {
    template <typename bit_type>
    using policy = WastefulRows<bit_type>;

    static std::string name() {
        return "ModeWastefulRows";
    }
};

struct HalfPackedRowsMode {
    template <typename bit_type>
    using policy = HalfPackedRows<bit_type>;

    static std::string name() {
        return "ModeHalfPackedRows";
    }
};

struct FullyPackedRowsMode {
    template <typename bit_type>
    using policy = FullyPackedRows<bit_type>;

    static std::string name() {
        return "ModeFullyPackedRows";
    }
};

struct AdderMode {
    template <typename bit_type>
    using policy = FullyPackedRows<bit_type>;

    static std::string name() {
        return "ModeAdder";
    }
};

struct FujitaMode {
    template <typename bit_type>
    using policy = BitColumns<bit_type>;

    static std::string name() {
        return "ModeFujita";
    }
};

} // namespace algorithms

#endif // ALGORITHMS_BITWISE_BIT_MODES_HPP