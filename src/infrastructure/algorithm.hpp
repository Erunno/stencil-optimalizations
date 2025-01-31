#ifndef INFRASTRUCTURE_ALGORITHM_HPP
#define INFRASTRUCTURE_ALGORITHM_HPP

#include <cstddef>
#include <iomanip>
#include <memory>
#include <string>

#include "experiment_params.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include "./colors.hpp"

namespace infrastructure {

template <int Dims, typename ElementType>
class Algorithm {
  public:
    using size_type = std::size_t;
    using DataGrid = Grid<Dims, ElementType>;

    Algorithm() = default;
    virtual ~Algorithm() = default;

    Algorithm(const Algorithm&) = delete;
    Algorithm& operator=(const Algorithm&) = delete;

    Algorithm(Algorithm&&) = default;
    Algorithm& operator=(Algorithm&&) = default;

    /**
     * @brief Sets the input data and formats it if necessary.
     *
     * @param data The input data grid.
     */
    virtual void set_and_format_input_data(const DataGrid& data) = 0;

    /**
     * @brief Initializes data structures and copies data to the device if it is a CUDA algorithm.
     */
    virtual void initialize_data_structures() = 0;

    /**
     * @brief Runs the algorithm for a specified number of iterations.
     *
     * @param iterations The number of iterations to run.
     */
    virtual void run(size_type iterations) = 0;

    /**
     * @brief Copies data back to the host and finalizes it.
     */
    virtual void finalize_data_structures() = 0;

    /**
     * @brief Fetches the result of the algorithm and transforms it back to a grid format.
     *
     * @return DataGrid The result of the algorithm.
     */
    virtual DataGrid fetch_result() = 0;

    /**
     * @brief Returns the number of iterations actually performed by the algorithm.
     * This is useful when the algorithm is stopped prematurely due to params.max_runtime_seconds.
     *
     * @return std::size_t The number of iterations actually performed.
     */
    virtual std::size_t actually_performed_iterations() const = 0;

    /**
     * @brief Sets the experiment parameters.
     *
     * @param params The experiment parameters.
     */
    virtual void set_params(const ExperimentParams& params) {
        this->params = params;
    }

  protected:
    ExperimentParams params;
};

struct TimeReport {
    static constexpr double INVALID = -1;

    double set_and_format_input_data = INVALID;
    double initialize_data_structures = INVALID;
    double run = INVALID;
    double finalize_data_structures = INVALID;
    double fetch_result = INVALID;

    std::size_t actually_performed_iterations = 0;

    double runtime_per_iteration() const {
        if (run == INVALID || actually_performed_iterations == 0) {
            return INVALID;
        }

        return run / actually_performed_iterations;
    }

    std::string pretty_print() const {
        std::string title_color = c::time_report_title();
        std::string reset_color = c::reset_color();

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += pretty_print_line("  set_and_format_input_data:  ", set_and_format_input_data);
        result += pretty_print_line("  initialize_data_structures: ", initialize_data_structures);
        result += pretty_print_line("  run:                        ", run);
        result += pretty_number_of_iterations(actually_performed_iterations);
        result += pretty_print_line("  finalize_data_structures:   ", finalize_data_structures);
        result += pretty_print_line("  fetch_result:               ", fetch_result) + reset_color;
        // clang-format on

        return result;
    }

    std::string pretty_print_speedup(const TimeReport& bench) const {
        std::string title_color = c::time_report_title();
        std::string reset_color = c::reset_color();

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += pretty_print_speedup_line("  set_and_format_input_data:  ", set_and_format_input_data, bench.set_and_format_input_data);
        result += pretty_print_speedup_line("  initialize_data_structures: ", initialize_data_structures, bench.initialize_data_structures);
        result += pretty_print_speedup_line("  run:                        ", run, bench.run, 
            actually_performed_iterations, bench.actually_performed_iterations);
        result += pretty_number_of_iterations_speedup(actually_performed_iterations, bench);
        result += pretty_print_speedup_line("  finalize_data_structures:   ", finalize_data_structures, bench.finalize_data_structures);
        result += pretty_print_speedup_line("  fetch_result:               ", fetch_result, bench.fetch_result) + reset_color;
        // clang-format on

        return result;
    }

  private:
    std::string speedup_str(double bench, double time) const {
        std::string positive_color = c::time_report_positive();
        std::string negative_color = c::time_report_negative();
        std::string reset_color = c::reset_color();

        double speedup = bench / time;

        std::stringstream speedup_s;
        speedup_s << std::fixed << std::setprecision(2) << speedup << +" x";

        if (speedup > 1 || speedup_s.str() == "1.00") {
            return positive_color + speedup_s.str() + reset_color;
        }
        else {
            speedup_s << " (" << std::fixed << std::setprecision(2) << 1 / speedup << " x)";
            return negative_color + speedup_s.str() + reset_color;
        }
    }

    std::string pretty_print_line(const std::string& label, double time) const {
        if (time == INVALID) {
            return "";
        }

        std::string labels_color = c::time_report_labels();
        std::string time_color = c::time_report_time();
        std::string reset_color = c::reset_color();

        return labels_color + label + time_color + std::to_string(time) + " ms" + reset_color + "\n";
    }

    std::string pretty_print_speedup_line(const std::string& label, double time, double bench, std::size_t measured_alg_iters = 0, std::size_t bench_iters = 0) const {
        if (time == INVALID || bench == INVALID) {
            return "";
        }

        std::string labels_color = c::time_report_labels();
        std::string time_color = c::time_report_time();
        std::string reset_color = c::reset_color();

        double corrected_bench_time = bench;
        std::string correction = "";

        if (measured_alg_iters != bench_iters) {
            corrected_bench_time = bench * measured_alg_iters / bench_iters;
            correction = c::time_report_info() + " (estimated)" + reset_color;
        }

        return labels_color + label + time_color + std::to_string(time) + " ms ~ " + speedup_str(corrected_bench_time, time) +
               reset_color + correction + "\n";
    }

    std::string pretty_number_of_iterations(std::size_t iterations) const {
        std::string labels_color = c::time_report_sublabels();
        std::string time_color = c::time_report_time();
        std::string reset_color = c::reset_color();

        auto line = labels_color + "   performed iters:  " + time_color + std::to_string(iterations) + reset_color + "\n";
        line +=     labels_color + "   runtime per iter: " + time_color + std::to_string(runtime_per_iteration()) + " ms" + reset_color + "\n";

        return line;
    }

    std::string pretty_number_of_iterations_speedup(std::size_t iterations, const TimeReport& bench) const {
        std::string labels_color = c::time_report_sublabels();
        std::string time_color = c::time_report_time();
        std::string reset_color = c::reset_color();

        std::string line = labels_color + "   performed iters:  " + time_color + std::to_string(iterations) + reset_color + "\n";
        line +=            labels_color + "   runtime per iter: " + time_color + std::to_string(runtime_per_iteration()) + 
            " ms ~ " + speedup_str(bench.runtime_per_iteration(), runtime_per_iteration()) + reset_color + "\n";

        return line;
    }
};

template <int Dims, typename ElementType>
class TimedAlgorithm : public Algorithm<Dims, ElementType> {
  public:
    using size_type = std::size_t;
    using DataGrid = Grid<Dims, ElementType>;

    TimedAlgorithm(Algorithm<Dims, ElementType>* algorithm) : algorithm(algorithm) {
    }

    void set_and_format_input_data(const DataGrid& data) override {
        time_report.set_and_format_input_data = Timer::measure_ms([&]() { algorithm->set_and_format_input_data(data); });
    }

    void initialize_data_structures() override {
        time_report.initialize_data_structures = Timer::measure_ms([&]() { algorithm->initialize_data_structures(); });
    }

    void run(size_type iterations) override {
        time_report.run = Timer::measure_ms([&]() { algorithm->run(iterations); });
        time_report.actually_performed_iterations = algorithm->actually_performed_iterations();
    }

    void finalize_data_structures() override {
        time_report.finalize_data_structures = Timer::measure_ms([&]() { algorithm->finalize_data_structures(); });
    }

    DataGrid fetch_result() override {
        DataGrid result;

        time_report.fetch_result = Timer::measure_ms([&]() { result = algorithm->fetch_result(); });

        return result;
    }

    std::size_t actually_performed_iterations() const override {
        return algorithm->actually_performed_iterations();
    }

    TimeReport get_time_report() {
        return time_report;
    }

    void set_params(const ExperimentParams& params) {
        algorithm->set_params(params);
    }

  private:
    TimeReport time_report;
    Algorithm<Dims, ElementType>* algorithm;
};

} // namespace infrastructure

#endif // INFRASTRUCTURE_ALGORITHM_HPP