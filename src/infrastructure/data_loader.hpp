#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <iostream>
#include <memory>
#include <random>

#include "experiment_params.hpp"
#include "gol-lexicon/lexicon.hpp"
#include "grid.hpp"

namespace infrastructure {

template <int Dims, typename ElementType>
class Loader {
  public:
    Loader() = default;
    virtual ~Loader() = default;

    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;

    Loader(Loader&&) = default;
    Loader& operator=(Loader&&) = default;

    virtual Grid<Dims, ElementType> load_data(const ExperimentParams& params) = 0;

    virtual std::unique_ptr<Grid<Dims, ElementType>> load_validation_data(const ExperimentParams& params) {
        (void)params;
        return nullptr;
    }
};

template <int Dims, typename ElementType>
class LoaderCtorBase {
  public:
    LoaderCtorBase() = default;
    virtual ~LoaderCtorBase() = default;

    LoaderCtorBase(const LoaderCtorBase&) = delete;
    LoaderCtorBase& operator=(const LoaderCtorBase&) = delete;

    LoaderCtorBase(LoaderCtorBase&&) = default;
    LoaderCtorBase& operator=(LoaderCtorBase&&) = default;

    virtual std::unique_ptr<Loader<Dims, ElementType>> create() = 0;
};

template <template <int Dims, typename ElementType> class LoaderType, int Dims, typename ElementType>
class LoaderCtor : public LoaderCtorBase<Dims, ElementType> {
  public:
    std::unique_ptr<Loader<Dims, ElementType>> create() override {
        return std::make_unique<LoaderType<Dims, ElementType>>();
    }
};

template <int Dims, typename ElementType>
class RandomOnesZerosDataLoader : public Loader<Dims, ElementType> {
  public:
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) override {
        Grid<Dims, ElementType> grid(params.grid_dimensions);

        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_int_distribution<int> dist(0, 1);

        auto grid_data = grid.data();

        for (std::size_t i = 0; i < grid.size(); ++i) {
            grid_data[i] = static_cast<ElementType>(dist(rng));
        }

        return grid;
    }
};

template <int Dims, typename ElementType>
class LexiconLoader : public Loader<Dims, ElementType> {
  public:
    Grid<Dims, ElementType> load_data(const ExperimentParams& params) {

        Grid<Dims, ElementType> grid(params.grid_dimensions);

        // Lexicon is not supported for general dimension

        return grid;
    }
};

template <typename ElementType>
class LexiconLoader<2, ElementType> : public Loader<2, ElementType> {
  public:
    Grid<2, ElementType> load_data(const ExperimentParams& params) {

        Grid<2, ElementType> grid(params.grid_dimensions);

        lexicon::Lexicon lexicon;
        lexicon.insert_patters(grid, params.pattern_expression);

        return grid;
    }
};

} // namespace infrastructure

#endif // DATA_LOADER_HPP
