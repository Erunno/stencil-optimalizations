#ifndef STATIC_SWITCH_HPP
#define STATIC_SWITCH_HPP

#include <tuple>
#include <utility>
#include <type_traits>
#include <array>

namespace templates {

// Helper template to hold compile-time options (Unchanged)
template <auto... Values>
struct options {
    static constexpr std::size_t size = sizeof...(Values);
    static constexpr std::array<std::common_type_t<decltype(Values)...>, size> values = {Values...};
    
    template <typename T>
    static constexpr bool contains(T value) {
        for (auto v : values) {
            if (v == value) return true;
        }
        return false;
    }
    
    template <typename T>
    static constexpr int index_of(T value) {
        for (std::size_t i = 0; i < size; ++i) {
            if (values[i] == value) return static_cast<int>(i);
        }
        return -1;
    }
};

// Main static_switch template
template <typename... OptionsPacks>
struct static_switch {
    static constexpr size_t NumPacks = sizeof...(OptionsPacks);

    // Entry point: Accepts any number of arguments to bypass greedy deduction issues
    template <typename... Args>
    static void call_with(Args&&... args) {
        static_assert(sizeof...(Args) == NumPacks + 2, 
            "Arguments provided must be: [Values...] followed by MatchCallback and FallbackCallback");

        // Pack arguments into a tuple for easy access
        auto args_tuple = std::forward_as_tuple(args...);

        // Extract callbacks (the last two arguments)
        auto& match_cb = std::get<NumPacks>(args_tuple);
        auto& fallback_cb = std::get<NumPacks + 1>(args_tuple);

        // Start recursion with 0th pack and empty matched values
        dispatch<0>(args_tuple, match_cb, fallback_cb);
    }

private:
    // Recursive dispatcher
    // CurrentPackIndex: Which OptionPack we are currently checking
    // MatchedValues...: The compile-time constants we have found so far
    template <std::size_t CurrentPackIndex, auto... MatchedValues, 
              typename Tuple, typename MatchCB, typename FallbackCB>
    static void dispatch(Tuple&& args, MatchCB&& match_cb, FallbackCB&& fallback_cb) {
        
        // Base Case: If we have matched all packs, call the callback with the accumulated template args
        if constexpr (CurrentPackIndex == NumPacks) {
            match_cb.template operator()<MatchedValues...>();
        } else {
            // Get the runtime value for the current position
            auto runtime_value = std::get<CurrentPackIndex>(args);
            
            // Get the type of the current OptionsPack
            using CurrentOptions = std::tuple_element_t<CurrentPackIndex, std::tuple<OptionsPacks...>>;
            
            // Flag to track if we found a match in this pack
            bool found = false;

            // Try to match the runtime value against the compile-time options
            try_match_option<CurrentPackIndex, 0, MatchedValues...>(
                runtime_value, found, args, match_cb, fallback_cb
            );

            // If this level didn't find a match, trigger fallback
            if (!found) {
                fallback_cb();
            }
        }
    }

    // Helper to iterate over the options within a specific OptionsPack
    template <std::size_t CurrentPackIndex, std::size_t OptionIndex, auto... MatchedValues, 
              typename T, typename Tuple, typename MatchCB, typename FallbackCB>
    static void try_match_option(T runtime_value, bool& found, 
                                 Tuple&& args, MatchCB&& match_cb, FallbackCB&& fallback_cb) {
        
        using CurrentOptions = std::tuple_element_t<CurrentPackIndex, std::tuple<OptionsPacks...>>;

        if constexpr (OptionIndex < CurrentOptions::size) {
            // Get the candidate value at compile time
            constexpr auto candidate = CurrentOptions::values[OptionIndex];

            if (runtime_value == candidate) {
                found = true;
                // RECURSION STEP: 
                // Move to PackIndex + 1
                // Add 'candidate' to the MatchedValues list
                dispatch<CurrentPackIndex + 1, MatchedValues..., candidate>(
                    std::forward<Tuple>(args), 
                    std::forward<MatchCB>(match_cb), 
                    std::forward<FallbackCB>(fallback_cb)
                );
            } else {
                // Runtime check failed, try the next option in this pack
                try_match_option<CurrentPackIndex, OptionIndex + 1, MatchedValues...>(
                    runtime_value, found, args, match_cb, fallback_cb
                );
            }
        }
    }
};

} // namespace templates

#endif // STATIC_SWITCH_HPP