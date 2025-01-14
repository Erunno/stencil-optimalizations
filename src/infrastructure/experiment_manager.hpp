#ifndef INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP
#define INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP

#include "../algorithms/an5d/an5d-cpu-alg.hpp"
#include "../algorithms/cpu-bitwise-cols-macro/gol_cpu_bitwise_cols_macro.hpp"
#include "../algorithms/cpu-bitwise-cols/gol_cpu_bitwise_cols.hpp"
#include "../algorithms/cpu-naive/gol_cpu_naive.hpp"
#include "../algorithms/cuda-naive-bitwise/gol_cuda_naive_bitwise.hpp"
#include "../algorithms/cuda-naive-local/gol_cuda_naive_local.hpp"
#include "../algorithms/cuda-naive/gol_cuda_naive.hpp"
#include "./data_loader.hpp"
#include "algorithm.hpp"
#include "algorithm_repository.hpp"
#include "data_loader.hpp"
#include "experiment_params.hpp"
#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>

#include "../debug_utils/diff_grid.hpp"
#include "../debug_utils/pretty_print.hpp"
#include "grid.hpp"
using namespace debug_utils;
namespace alg = algorithms;
namespace cuda_naive_local = algorithms::cuda_naive_local;

namespace infrastructure {

class ExperimentManager {
    template <int Dims, typename ElementType>
    using grid_ptr = std::unique_ptr<Grid<Dims, ElementType>>;
    template <typename col_type>
    using MacroOps = algorithms::MacroBitOperations<col_type>;

    enum class AlgMode {
        Timed = 0,
        NotTimed = 1,
    };

  public:
    ExperimentManager() {

        auto _2d_repo = _algs_repos.get_repository<2, char>();

        // CPU

        _2d_repo->register_algorithm<alg::GoLCpuNaive>("gol-cpu-naive");

        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<16>>("gol-cpu-bitwise-cols-16");
        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<32>>("gol-cpu-bitwise-cols-32");
        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<64>>("gol-cpu-bitwise-cols-64");

        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<16, MacroOps>>("gol-cpu-bitwise-cols-macro-16");
        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<32, MacroOps>>("gol-cpu-bitwise-cols-macro-32");
        _2d_repo->register_algorithm<alg::GoLCpuBitwiseCols<64, MacroOps>>("gol-cpu-bitwise-cols-macro-64");

        // CUDA

        _2d_repo->register_algorithm<alg::GoLCudaNaive>("gol-cuda-naive");

        _2d_repo->register_algorithm<alg::GoLCudaNaiveBitwise<16>>("gol-cuda-naive-bitwise-cols-16");
        _2d_repo->register_algorithm<alg::GoLCudaNaiveBitwise<32>>("gol-cuda-naive-bitwise-cols-32");
        _2d_repo->register_algorithm<alg::GoLCudaNaiveBitwise<64>>("gol-cuda-naive-bitwise-cols-64");

        _2d_repo->register_algorithm<cuda_naive_local::GoLCudaNaiveLocal<16>>("gol-cuda-naive-local-16");
        _2d_repo->register_algorithm<cuda_naive_local::GoLCudaNaiveLocal<32>>("gol-cuda-naive-local-32");
        _2d_repo->register_algorithm<cuda_naive_local::GoLCudaNaiveLocal<64>>("gol-cuda-naive-local-64");

        // AN5D

        _2d_repo->register_algorithm<alg::An5dAlg<32, alg::ExecModel::CPU>>("an5d-cpu-32");
        _2d_repo->register_algorithm<alg::An5dAlg<64, alg::ExecModel::CPU>>("an5d-cpu-64");

        _2d_repo->register_algorithm<alg::An5dAlg<32, alg::ExecModel::CUDA>>("an5d-cuda-32");
        _2d_repo->register_algorithm<alg::An5dAlg<64, alg::ExecModel::CUDA>>("an5d-cuda-64");
    }

    void run(const ExperimentParams& params) {
        print_basic_param_info(params);

        _algs_repos.for_each([&](auto& repo) {
            if (repo.has_algorithm(params.algorithm_name)) {
                run_experiment(repo, params);
            }
        });
    }

  private:
    template <int Dims, typename ElementType>
    void run_experiment(AlgorithmRepository<Dims, ElementType>& repo, const ExperimentParams& params) {

        auto loader = fetch_loader<Dims, ElementType>(params);

        std::cout << title_color << "Loading data... " << param_color << "             " << params.data_loader_name
                  << ((params.data_loader_name == "lexicon")
                          ? title_color + " with pattern " + param_color + params.pattern_expression
                          : "")
                  << reset_color << std::endl;

        Timer t;

        Grid<Dims, ElementType> data;

        auto ms = t.measure([&]() { data = loader->load_data(params); });
        std::cout << label_color << "  Data loaded in " << param_color << ms << " ms" << reset_color << std::endl;

        auto alg = repo.fetch_algorithm(params.algorithm_name);
        TimeReport bench_report;

        if (params.measure_speedup) {
            bench_report = measure_speedup(params, data);
        }

        std::cout << title_color << "Running experiment...        " << param_color << params.algorithm_name
                  << reset_color << std::endl;
        auto [result, time_report] = perform_alg<AlgMode::Timed>(*alg, data, params);

        if (params.measure_speedup) {
            std::cout << time_report.pretty_print_speedup(bench_report) << std::endl;
        }
        else {
            std::cout << time_report.pretty_print() << std::endl;
        }

        if (params.validate) {
            validate(data, *result.get(), *loader.get(), params);
        }
    }

    template <int Dims, typename ElementType>
    void validate(const Grid<Dims, ElementType>& original, const Grid<Dims, ElementType>& result,
                  Loader<Dims, ElementType>& loader, const ExperimentParams& params) {

        std::cout << title_color << "Validating result... " << param_color << params.validation_algorithm_name
                  << reset_color << std::endl;

        auto validation_data = loader.load_validation_data(params);

        if (validation_data == nullptr) {
            auto repo = _algs_repos.get_repository<Dims, ElementType>();
            auto alg = repo->fetch_algorithm(params.validation_algorithm_name);

            validation_data = perform_alg<AlgMode::NotTimed>(*alg, original, params);
        }

        if (validation_data->equals(result)) {
            std::cout << "\033[32mValidation successful\033[0m" << std::endl;
        }
        else {
            std::cout << "\033[31mValidation failed\033[0m" << std::endl;

            if (params.print_validation_diff) {
                auto diff = debug_utils::diff(*validation_data.get(), result);
                std::cout << "Diff: \n" << diff << std::endl;
            }
        }
    }

    template <int Dim, typename ElementType>
    TimeReport measure_speedup(const ExperimentParams& params, const Grid<Dim, ElementType>& init_data) {
        std::cout << title_color << "Measuring bench algorithm... " << param_color
                  << params.speedup_bench_algorithm_name << reset_color << std::endl;

        auto repo = _algs_repos.get_repository<Dim, ElementType>();
        auto alg = repo->fetch_algorithm(params.speedup_bench_algorithm_name);

        auto [result, time_report] = perform_alg<AlgMode::Timed>(*alg, init_data, params);

        return time_report;
    }

    template <AlgMode mode, int Dims, typename ElementType>
    auto perform_alg(Algorithm<Dims, ElementType>& alg, const Grid<Dims, ElementType>& init_data,
                     const ExperimentParams& params) {

        TimedAlgorithm<Dims, ElementType> timed_alg(&alg);

        timed_alg.set_params(params);

        timed_alg.set_and_format_input_data(init_data);
        timed_alg.initialize_data_structures();
        timed_alg.run(params.iterations);
        timed_alg.finalize_data_structures();

        std::unique_ptr<Grid<Dims, ElementType>> result_ptr = nullptr;

        if (params.validate) {
            auto result = timed_alg.fetch_result();
            result_ptr = std::make_unique<Grid<Dims, ElementType>>(result);
        }

        auto time_report = timed_alg.get_time_report();

        if constexpr (mode == AlgMode::Timed) {
            return std::make_tuple(std::move(result_ptr), time_report);
        }
        else {
            return result_ptr;
        }
    }

    template <int Dims, typename ElementType>
    std::unique_ptr<Loader<Dims, ElementType>> fetch_loader(const ExperimentParams& params) {
        LoaderCtor<RandomOnesZerosDataLoader, Dims, ElementType> random_loader;
        LoaderCtor<LexiconLoader, Dims, ElementType> lexicon_loader;

        auto loaderCtor = std::map<std::string, LoaderCtorBase<Dims, ElementType>*>{
            {"random-ones-zeros", &random_loader},
            {"lexicon", &lexicon_loader},
        }[params.data_loader_name];

        return loaderCtor->create();
    }

    void print_basic_param_info(const ExperimentParams& params) const {
        std::cout << title_color << "Experiment parameters:" << reset_color << std::endl;
        std::cout << label_color << "  Grid dimensions: " << param_color << print_grid_dims(params.grid_dimensions)
                  << reset_color << std::endl;
        std::cout << label_color << "  Iterations:      " << param_color << print_num(params.iterations) << reset_color
                  << std::endl;

        std::cout << std::endl;
    }

    std::string print_grid_dims(const std::vector<std::size_t>& dims) const {
        std::string result = "";
        for (auto&& dim : dims) {
            result += print_num(dim) + " x ";
        }
        result.pop_back();
        result.pop_back();
        return result;
    }

    std::string print_num(std::size_t num) const {
        std::string result = std::to_string(num);
        for (int i = result.size() - 3; i > 0; i -= 3) {
            result.insert(i, "'");
        }
        return result;
    }

    AlgorithmReposCollection<AlgRepoParams<2, char>, AlgRepoParams<3, char>> _algs_repos;

    std::string title_color = "\033[35m";
    std::string param_color = "\033[33m";
    std::string label_color = "\033[36m";
    std::string reset_color = "\033[0m";
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP
