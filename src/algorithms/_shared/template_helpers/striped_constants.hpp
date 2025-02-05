#ifndef ALGORITHMS_STRIPED_CONSTANTS_HPP
#define ALGORITHMS_STRIPED_CONSTANTS_HPP

namespace algorithms {

template <typename word_type, int BITS, int constant, int stripe_size>
struct ConstsWithBits {
    static constexpr word_type expanded = ConstsWithBits<word_type, BITS - stripe_size, constant, stripe_size>::expanded << stripe_size | constant;
};

template <typename word_type, int constant, int stripe_size>
struct ConstsWithBits<word_type, 0, constant, stripe_size> {
    static constexpr word_type expanded = 0;
};

template <typename word_type, int constant, int stripe_size>
struct Consts {
    static constexpr word_type expanded = ConstsWithBits<word_type, sizeof(word_type) * 8, constant, stripe_size>::expanded;
};

} // namespace algorithms

#endif // ALGORITHMS_STRIPED_CONSTANTS_HPP