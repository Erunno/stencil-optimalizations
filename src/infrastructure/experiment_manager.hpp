#ifndef INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP
#define INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP

#include "../algorithms/_shared/bitwise/bitwise-ops/cpu-ops-interface.hpp"
#include "../algorithms/cpu-bitwise-general/gol_cpu_bitwise_general.hpp"
#include "../algorithms/cpu-bitwise-general-naive/gol_cpu_bitwise_general_naive.hpp"
#include "../algorithms/cpu-naive/gol_cpu_naive.hpp"
#include "../algorithms/cuda-naive-bitwise/gol_cuda_naive_bitwise.hpp"
#include "../algorithms/cuda-naive/gol_cuda_naive.hpp"
#include "../algorithms/cuda-memcpy/cuda_memcpy_bench.hpp"
#include "../algorithms/cuda-naive-local-one-cell/cuda_local_one_cell.hpp"
#include "./data_loader.hpp"
#include "../algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/register-all-algs.hpp"
#include "algorithm.hpp"
#include "algorithm_repository.hpp"
#include "data_loader.hpp"
#include "experiment_params.hpp"
#include <cstddef>
#include <cstdint>
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
namespace cuda_local_one_cell = algorithms::cuda_local_one_cell;

using state_in_64_bits_t = std::uint64_t;
using state_in_32_bits_t = std::uint32_t;

namespace infrastructure {

template <typename grid_cell_t>
class ExperimentManager {
    template <int Dims, typename ElementType>
    using grid_ptr = std::unique_ptr<Grid<Dims, ElementType>>;

    template <typename word_type>
    using ColsMacroOps = algorithms::MacroColsOperations<word_type>;

    template <typename word_type>
    using TilesMacroOps = algorithms::MacroTilesOperations<word_type>;

    template <typename word_type>
    using ColsTemplatedOps = algorithms::BitwiseColsTemplatedOps<word_type>;

    enum class AlgMode {
        Timed = 0,
        NotTimed = 1,
    };

  public:
    ExperimentManager() {

        auto _2d_repo = _algs_repos.template get_repository<2, grid_cell_t>();

        // CPU

        // baseline
        _2d_repo-> template register_algorithm<alg::GoLCpuNaive<grid_cell_t>>("gol-cpu-naive");

        // this is naive implementation which is ignorant of the underling bitwise encoding
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 16, alg::BitColumnsMode>>("gol-cpu-bitwise-cols-naive-16");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 32, alg::BitColumnsMode>>("gol-cpu-bitwise-cols-naive-32");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 64, alg::BitColumnsMode>>("gol-cpu-bitwise-cols-naive-64");

        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 16, alg::BitTileMode>>("gol-cpu-bitwise-tiles-naive-16");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 32, alg::BitTileMode>>("gol-cpu-bitwise-tiles-naive-32");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwiseNaive<grid_cell_t, 64, alg::BitTileMode>>("gol-cpu-bitwise-tiles-naive-64");

        // this is avare that the encoding are bit columns and uses templates to take advantage of that
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 16, ColsTemplatedOps>>("gol-cpu-bitwise-cols-16");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 32, ColsTemplatedOps>>("gol-cpu-bitwise-cols-32");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 64, ColsTemplatedOps>>("gol-cpu-bitwise-cols-64");

        // this is avare that the encoding are bit columns and uses generated macro
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 16, ColsMacroOps>>("gol-cpu-bitwise-cols-macro-16");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 32, ColsMacroOps>>("gol-cpu-bitwise-cols-macro-32");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 64, ColsMacroOps>>("gol-cpu-bitwise-cols-macro-64");

        // this is avare that the encoding are bit tiles and uses generated macro
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 16, TilesMacroOps>>("gol-cpu-bitwise-tiles-macro-16");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 32, TilesMacroOps>>("gol-cpu-bitwise-tiles-macro-32");
        _2d_repo-> template register_algorithm<alg::GoLCpuBitwise<grid_cell_t, 64, TilesMacroOps>>("gol-cpu-bitwise-tiles-macro-64");

        // CUDA

        _2d_repo-> template register_algorithm<alg::GoLCudaNaive<grid_cell_t>>("gol-cuda-naive");

        _2d_repo-> template register_algorithm<alg::CudaMemcpy<grid_cell_t>>("cuda-memcpy");

        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 16, alg::BitColumnsMode>>("gol-cuda-naive-bitwise-cols-16");
        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 32, alg::BitColumnsMode>>("gol-cuda-naive-bitwise-cols-32");
        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 64, alg::BitColumnsMode>>("gol-cuda-naive-bitwise-cols-64");

        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 16, alg::BitTileMode>>("gol-cuda-naive-bitwise-tiles-16");
        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 32, alg::BitTileMode>>("gol-cuda-naive-bitwise-tiles-32");
        _2d_repo-> template register_algorithm<alg::GoLCudaNaiveBitwise<grid_cell_t, 64, alg::BitTileMode>>("gol-cuda-naive-bitwise-tiles-64");

        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 16, alg::BitColumnsMode>>("gol-cuda-local-one-cell-cols-16");
        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 32, alg::BitColumnsMode>>("gol-cuda-local-one-cell-cols-32");
        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 64, alg::BitColumnsMode>>("gol-cuda-local-one-cell-cols-64");

        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 16, alg::BitTileMode>>("gol-cuda-local-one-cell-16--bit-tiles");
        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 32, alg::BitTileMode>>("gol-cuda-local-one-cell-32--bit-tiles");
        _2d_repo-> template register_algorithm<cuda_local_one_cell::GoLCudaLocalOneCell<grid_cell_t, 64, alg::BitTileMode>>("gol-cuda-local-one-cell-64--bit-tiles");

        // related work algorithms

        algorithms::rel_work::Efficient_simulation_execution_of_cellular_automata_on_GPU::register_all_algs(*_2d_repo);
    }

    void run(const ExperimentParams& params) {
        // print_basic_param_info(params);

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

        std::cout << c::title_color() << "Loading data... " << c::value_color() << "             " << params.data_loader_name
                  << ((params.data_loader_name == "lexicon") ? c::title_color() + " with pattern " + c::value_color() + params.pattern_expression: "") 
                  << c::reset_color() << std::endl;
        
        Grid<Dims, ElementType> data;

        auto milli_secs  = Timer::measure_ms([&]() { data = loader->load_data(params); });
        std::cout << c::label_color() << "  Data loaded in   " << c::value_color() << milli_secs << " ms" << c::reset_color() << std::endl << std::endl;

        TimeReport bench_report;

        if (params.warmup_rounds != 0 && params.measure_speedup) {
            auto alg = repo.fetch_algorithm(params.speedup_bench_algorithm_name);
            warm_up(*alg, data, params);
        }

        if (params.measure_speedup) {
            bench_report = measure_speedup(params, data);
        }

        if (params.warmup_rounds != 0) {
            auto alg = repo.fetch_algorithm(params.algorithm_name);
            warm_up(*alg, data, params);
        }

        auto [result, iterations] = run_measurements(data, bench_report, repo, params);

        if (params.validate) {
            validate(data, *result.get(), iterations, *loader.get(), params);
        }
    }

    template <int Dims, typename ElementType>
    void validate(const Grid<Dims, ElementType>& original, const Grid<Dims, ElementType>& result, std::size_t iterations,
                  Loader<Dims, ElementType>& loader, const ExperimentParams& params) {

        std::cout << c::title_color() << "Validating result... " << c::value_color() << params.validation_algorithm_name
                  << c::reset_color() << std::endl;

        auto validation_data = loader.load_validation_data(params);

        if (validation_data == nullptr) {
            auto repo = _algs_repos.template get_repository<Dims, ElementType>();
            auto alg = repo->fetch_algorithm(params.validation_algorithm_name);

            auto validation_params = params;
            validation_params.iterations = iterations;
            validation_params.max_runtime_seconds = infrastructure::StopWatch::MAX_TIME;

            validation_data = perform_alg<AlgMode::NotTimed>(*alg, original, validation_params);
        }

        if (validation_data->equals(result)) {
            std::cout << c::success_color() << "Validation successful" << c::reset_color() << std::endl;
        }
        else {
            std::cout << c::error_color() << "Validation failed" << c::reset_color() << std::endl;

            if (params.print_validation_diff) {
                auto diff = debug_utils::diff(*validation_data.get(), result);
                std::cout << "Diff: \n" << diff << std::endl;
            }
        }
    }

    template <int Dim, typename ElementType>
    TimeReport measure_speedup(const ExperimentParams& params, const Grid<Dim, ElementType>& init_data) {
        std::cout << c::title_color() << "Measuring bench algorithm... " << c::value_color()
                  << params.speedup_bench_algorithm_name << c::value_color() << std::endl;

        auto repo = _algs_repos.template get_repository<Dim, ElementType>();
        auto alg = repo->fetch_algorithm(params.speedup_bench_algorithm_name);

        auto [result, time_report] = perform_alg<AlgMode::Timed>(*alg, init_data, params);

        return time_report;
    }

    template <int Dims, typename ElementType>
    void warm_up(Algorithm<Dims, ElementType>& alg, const Grid<Dims, ElementType>& init_data,
                 const ExperimentParams& params) {

        for (std::size_t i = 0; i < params.warmup_rounds; i++) {
            std::cout << c::title_color() << "Warming up... " << c::value_color()  << i + 1 << " / " <<  params.warmup_rounds << c::reset_color() << std::endl;
            perform_alg<AlgMode::NotTimed>(alg, init_data, params);
            std::cout << c::line_up();
        }

        std::cout << std::endl << std::endl;
    }


    template <int Dims, typename ElementType>
    auto run_measurements(Grid<Dims, ElementType>& data, TimeReport& bench_report, AlgorithmRepository<Dims, ElementType>& repo, const ExperimentParams& params) {
        std::cout << c::title_color() << "Running experiment...        " << c::value_color() << params.algorithm_name
                  << c::reset_color() << std::endl << std::endl;

        grid_ptr<Dims, ElementType> last_result = nullptr;
        std::size_t iterations = 0;

        for (std::size_t i = 0; i < params.measurement_rounds; i++) {
            std::cout << c::title_color() << "Measurement round " << c::value_color() << i + 1 << " / " << params.measurement_rounds
                      << c::reset_color() << std::endl;

            auto alg = repo.fetch_algorithm(params.algorithm_name);
            auto [result, time_report] = perform_alg<AlgMode::Timed>(*alg, data, params);
            
            last_result = std::move(result);
            iterations = time_report.actually_performed_iterations;

            if (params.measure_speedup) {
                std::cout << time_report.pretty_print_speedup(bench_report) << std::endl;
            }
            else {
                std::cout << time_report.pretty_print() << std::endl;
            }
        }

        return std::make_tuple(std::move(last_result), iterations);
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
        LoaderCtor<AlwaysChangingSpaceLoader, Dims, ElementType> always_changing_loader;
        LoaderCtor<ZerosLoader, Dims, ElementType> zeros_loader;


        auto loaderCtor = std::map<std::string, LoaderCtorBase<Dims, ElementType>*>{
            {"random-ones-zeros", &random_loader},
            {"lexicon", &lexicon_loader},
            {"always-changing", &always_changing_loader},
            {"zeros", &zeros_loader},
        }[params.data_loader_name];

        return loaderCtor->create();
    }

    void print_basic_param_info(const ExperimentParams& params) const {
        std::cout << c::title_color() << "Experiment parameters:" << c::reset_color() << std::endl;
        std::cout << c::label_color() << "  Grid dimensions: " << c::value_color() << print_grid_dims(params.grid_dimensions)
                  << c::reset_color() << std::endl;
        std::cout << c::label_color() << "  Iterations:      " << c::value_color() << print_num(params.iterations) << c::reset_color()
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

    AlgorithmReposCollection<AlgRepoParams<2, grid_cell_t>, AlgRepoParams<3, grid_cell_t>> _algs_repos;

};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_MANAGER_HPP