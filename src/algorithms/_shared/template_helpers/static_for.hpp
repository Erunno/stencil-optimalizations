#ifndef STATIC_FOR_HPP
#define STATIC_FOR_HPP

#include <cstddef>
#include <utility>

namespace templates {

template <std::size_t I, std::size_t N>
struct static_for {
    template <typename Functor>
    static void run(Functor&& f) {
        f.template operator()<I>();
        static_for<I + 1, N>::run(std::forward<Functor>(f));
    }
};

template <std::size_t N>
struct static_for<N, N> {
    template <typename Functor>
    static void run(Functor&&) {
    }
};

} // namespace templates

// usage

// template <std::size_t N>
// void test_f() {
//     std::cout << "N: " << N << std::endl;
// }

// ...

// static_for<0, 64>::run(
//     []<std::size_t I>() {
//         test_f<I>();
//     }
// );

#endif // STATIC_FOR_HPP
