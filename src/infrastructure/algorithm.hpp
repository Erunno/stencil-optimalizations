#ifndef INFRASTRUCTURE_ALGORITHM_HPP
#define INFRASTRUCTURE_ALGORITHM_HPP

#include <cstddef>
#include <iomanip>
#include <memory>
#include <string>

#include "../algorithms/an5d/an5d_cuda_timer.hpp"
#include "experiment_params.hpp"
#include "grid.hpp"
#include "timer.hpp"

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
     * @brief Sets the experiment parameters.
     *
     * @param params The experiment parameters.
     */
    void set_params(const ExperimentParams& params) {
        this->params = params;
    }

    virtual bool is_an5d_cuda_alg() const {
        return false;
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

    std::string pretty_print() const {
        std::string title_color = "\033[1;34m";
        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[32m";
        std::string reset_color = "\033[0m";

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += pretty_print_line("  set_and_format_input_data:  ", set_and_format_input_data);
        result += pretty_print_line("  initialize_data_structures: ", initialize_data_structures);
        result += pretty_print_line("  run:                        ", run);
        result += pretty_print_line("  finalize_data_structures:   ", finalize_data_structures);
        result += pretty_print_line("  fetch_result:               ", fetch_result);
        // clang-format on

        return result;
    }

    std::string pretty_print_speedup(const TimeReport& bench) const {
        std::string title_color = "\033[1;34m";
        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[34m";
        std::string reset_color = "\033[0m";

        // clang-format off
        std::string result = title_color + "Time report:\n";
        result += pretty_print_speedup_line("  set_and_format_input_data:  ", set_and_format_input_data, bench.set_and_format_input_data);
        result += pretty_print_speedup_line("  initialize_data_structures: ", initialize_data_structures, bench.initialize_data_structures);
        result += pretty_print_speedup_line("  run:                        ", run, bench.run);
        result += pretty_print_speedup_line("  finalize_data_structures:   ", finalize_data_structures, bench.finalize_data_structures);
        result += pretty_print_speedup_line("  fetch_result:               ", fetch_result, bench.fetch_result) + reset_color;
        // clang-format on

        return result;
    }

  private:
    std::string speedup_str(double bench, double time) const {
        std::string positive_color = "\033[32m";
        std::string negative_color = "\033[31m";
        std::string reset_color = "\033[0m";

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

        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[32m";
        std::string reset_color = "\033[0m";

        return labels_color + label + time_color + std::to_string(time) + "s" + reset_color + "\n";
    }

    std::string pretty_print_speedup_line(const std::string& label, double time, double bench) const {
        if (time == INVALID || bench == INVALID) {
            return "";
        }

        std::string labels_color = "\033[1;33m";
        std::string time_color = "\033[32m";
        std::string reset_color = "\033[0m";

        return labels_color + label + time_color + std::to_string(time) + "s ~ " + speedup_str(bench, time) +
               reset_color + "\n";
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
        Timer t;

        time_report.set_and_format_input_data = t.measure([&]() { algorithm->set_and_format_input_data(data); });
    }

    void initialize_data_structures() override {
        Timer t;

        time_report.initialize_data_structures = t.measure([&]() { algorithm->initialize_data_structures(); });
    }

    void run(size_type iterations) override {
        Timer t;

        time_report.run = t.measure([&]() { algorithm->run(iterations); });
    }

    void finalize_data_structures() override {
        Timer t;

        time_report.finalize_data_structures = t.measure([&]() { algorithm->finalize_data_structures(); });
    }

    DataGrid fetch_result() override {
        Timer t;

        DataGrid result;

        time_report.fetch_result = t.measure([&]() { result = algorithm->fetch_result(); });

        return result;
    }

    TimeReport get_time_report() {
        if (algorithm->is_an5d_cuda_alg()) {

            time_report.set_and_format_input_data = TimeReport::INVALID;
            time_report.initialize_data_structures = TimeReport::INVALID;

            time_report.run = algorithms::An5dCudaTimer::elapsed_time_s();

            time_report.finalize_data_structures = TimeReport::INVALID;
            time_report.fetch_result = TimeReport::INVALID;
        }

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
