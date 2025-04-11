#ifndef INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
#define INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP

#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cxxopts.hpp>
#include "colors.hpp"
#include "./timer.hpp"

namespace infrastructure {

enum class StreamingDirection {
    in_X = 0,
    in_Y = 1,
    NAIVE = 2,
};

class ExperimentParams {
  public:
    std::string algorithm_name;
    std::vector<std::size_t> grid_dimensions;
    std::size_t iterations;
    std::size_t max_runtime_seconds = infrastructure::StopWatch::MAX_TIME;

    std::string base_grid_encoding = "char"; // char, int

    std::size_t warmup_rounds = 0;
    std::size_t measurement_rounds = 1;

    std::string data_loader_name;
    std::string pattern_expression;

    bool measure_speedup = false;
    std::string speedup_bench_algorithm_name;

    bool validate = false;
    bool print_validation_diff = false;
    bool use_contracted_diff = false;

    std::string validation_algorithm_name = "gol-cpu-naive";

    bool animate_output = false;
    bool colorful = true;

    bool collect_touched_tiles_stats = false;

    std::size_t random_seed = 42;

    std::string tag = "";

    std::size_t state_bits_count = 64;

    std::size_t thread_block_size;

    std::size_t warp_dims_x;
    std::size_t warp_dims_y;
    std::size_t warp_tile_dims_x;
    std::size_t warp_tile_dims_y;

    std::size_t temporal_steps = 1;

    StreamingDirection streaming_direction;

    std::string pretty_print() {
      std::stringstream ss;

      std::string title_color = c::title_color();
      std::string label_color = c::label_color();
      std::string value_color = c::value_color();
      std::string reset_color = c::reset_color();

      ss << title_color << "Experiment Parameters:" << std::endl;

      ss << label_color << "  algorithm_name: " << value_color << algorithm_name << std::endl;
      ss << label_color << "  grid_dimensions: " << value_color << grid_dimensions[0] << "x" << grid_dimensions[1] << std::endl;
      ss << label_color << "  iterations: " << value_color << iterations << std::endl << c::extra_line_in_params();
      ss << label_color << "  base_grid_encoding: " << value_color << base_grid_encoding << std::endl << c::extra_line_in_params();
      ss << label_color << "  max_runtime_seconds: " << value_color << max_runtime_seconds << std::endl << c::extra_line_in_params();
      ss << label_color << "  warmup_rounds: " << value_color << warmup_rounds << std::endl;
      ss << label_color << "  measurement_rounds: " << value_color << measurement_rounds << std::endl << c::extra_line_in_params();
      ss << label_color << "  data_loader_name: " << value_color << data_loader_name << std::endl;
      ss << label_color << "  pattern_expression: " << value_color << pattern_expression << std::endl << c::extra_line_in_params();
      ss << label_color << "  measure_speedup: " << value_color << measure_speedup << std::endl;
      ss << label_color << "  speedup_bench_algorithm_name: " << value_color << speedup_bench_algorithm_name << std::endl << c::extra_line_in_params();
      ss << label_color << "  validate: " << value_color << validate << std::endl;
      ss << label_color << "  print_validation_diff: " << value_color << print_validation_diff << std::endl;
      ss << label_color << "  use_contracted_diff: " << value_color << use_contracted_diff << std::endl;
      ss << label_color << "  validation_algorithm_name: " << value_color << validation_algorithm_name << std::endl << c::extra_line_in_params();
      ss << label_color << "  animate_output: " << value_color << animate_output << std::endl;
      ss << label_color << "  colorful: " << value_color << colorful << std::endl << c::extra_line_in_params();
      ss << label_color << "  random_seed: " << value_color << random_seed << std::endl << c::extra_line_in_params();
      ss << label_color << "  tag: " << value_color << tag << std::endl << c::extra_line_in_params();
      ss << label_color << "  state_bits_count: " << value_color << state_bits_count << std::endl << c::extra_line_in_params();
      ss << label_color << "  thread_block_size: " << value_color << thread_block_size << std::endl << c::extra_line_in_params();
      ss << label_color << "  warp_dims_x: " << value_color << warp_dims_x << std::endl;
      ss << label_color << "  warp_dims_y: " << value_color << warp_dims_y << std::endl << c::extra_line_in_params();
      ss << label_color << "  warp_tile_dims_x: " << value_color << warp_tile_dims_x << std::endl;
      ss << label_color << "  warp_tile_dims_y: " << value_color << warp_tile_dims_y << std::endl << c::extra_line_in_params();
      ss << label_color << "  temporal_steps: " << value_color << temporal_steps << std::endl << c::extra_line_in_params();
      ss << label_color << "  collect_touched_tiles_stats: " << value_color << collect_touched_tiles_stats << std::endl << c::extra_line_in_params();
      
      ss << label_color << "  streaming_direction: " << value_color;
      switch (streaming_direction) {
        case StreamingDirection::in_X: ss << "in-x"; break;
        case StreamingDirection::in_Y: ss << "in-y"; break;
        case StreamingDirection::NAIVE: ss << "naive"; break;
      }
      ss << reset_color << std::endl;


      return ss.str();
    }
};

namespace opts {
// clang-format off
const std::string ALGORITHM_NAME               = "algorithm";
const std::string GRID_DIMENSIONS_X            = "grid-dimensions-x";
const std::string GRID_DIMENSIONS_Y            = "grid-dimensions-y";
const std::string ITERATIONS                   = "iterations";
const std::string MAX_RUNTIME_SECONDS          = "max-runtime-seconds";
const std::string WARMUP_ROUNDS                = "warmup-rounds";
const std::string MEASUREMENT_ROUNDS           = "measurement-rounds";
const std::string DATA_LOADER_NAME             = "data-loader";
const std::string PATTERN_EXPRESSION           = "pattern-expression";
const std::string MEASURE_SPEEDUP              = "measure-speedup";
const std::string SPEEDUP_BENCH_ALGORITHM_NAME = "speedup-bench-algorithm";
const std::string VALIDATE                     = "validate";
const std::string PRINT_VALIDATION_DIFF        = "print-validation-diff";
const std::string USE_CONTRACTED_DIFF          = "use-contracted-diff";
const std::string VALIDATION_ALGORITHM_NAME    = "validation-algorithm";
const std::string ANIMATE_OUTPUT               = "animate-output";
const std::string COLORFUL                     = "colorful";
const std::string RANDOM_SEED                  = "random-seed";
const std::string THREAD_BLOCK_SIZE            = "thread-block-size";
const std::string WARP_DIMS_X                  = "warp-dims-x";
const std::string WARP_DIMS_Y                  = "warp-dims-y";
const std::string WARP_TILE_DIMS_X             = "warp-tile-dims-x";
const std::string WARP_TILE_DIMS_Y             = "warp-tile-dims-y";
const std::string STREAMING_DIRECTION          = "streaming-direction";
const std::string STATE_BITS_COUNT             = "state-bits-count";
const std::string BASE_GRID_ENCODING           = "base-grid-encoding";
const std::string TAG                          = "tag";
const std::string COLLECT_TOUCHED_TILES_STATS  = "collect-touched-tiles-stats";
const std::string TEMPORAL_STEPS               = "temporal-steps";
// clang-format on
}

class ParamsParser {

  public:
    static ExperimentParams parse(int argc, char** argv) {
      cxxopts::Options optConfig("game of life", "");

      optConfig.add_options()
        ("help", "Show help")

        (opts::ALGORITHM_NAME, "Algorithm name",
          cxxopts::value<std::string>())

        (opts::GRID_DIMENSIONS_X, "Grid X dimension",
          cxxopts::value<std::size_t>())
        
        (opts::GRID_DIMENSIONS_Y, "Grid Y dimension",
          cxxopts::value<std::size_t>())
        
        (opts::ITERATIONS, "Number of iterations",
          cxxopts::value<std::size_t>())

        (opts::MAX_RUNTIME_SECONDS, "Maximum runtime in seconds",
          cxxopts::value<std::size_t>()->default_value(std::to_string(infrastructure::StopWatch::MAX_TIME)))

        (opts::WARMUP_ROUNDS, "Number of warmup rounds",
          cxxopts::value<std::size_t>()->default_value("0"))

        (opts::MEASUREMENT_ROUNDS, "Number of measurements",
          cxxopts::value<std::size_t>()->default_value("1"))
        
        (opts::DATA_LOADER_NAME, "Data loader",
          cxxopts::value<std::string>()->default_value("random-ones-zeros"))
        
        (opts::PATTERN_EXPRESSION, "Pattern expression",
          cxxopts::value<std::string>()->default_value(""))
        
        (opts::MEASURE_SPEEDUP, "Measure speedup",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::SPEEDUP_BENCH_ALGORITHM_NAME, "Speedup bench algorithm",
          cxxopts::value<std::string>()->default_value("gol-cpu-naive"))
        
        (opts::VALIDATE, "Validate",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::PRINT_VALIDATION_DIFF, "Print validation diff",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::USE_CONTRACTED_DIFF, "Use contracted diff for large grids",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::VALIDATION_ALGORITHM_NAME, "Validation algorithm",
          cxxopts::value<std::string>()->default_value("gol-cpu-naive"))
        
        (opts::ANIMATE_OUTPUT, "Animate output",
          cxxopts::value<bool>()->default_value("false"))
        
        (opts::COLORFUL, "Use colorful output",
          cxxopts::value<bool>()->default_value("true"))
        
        (opts::RANDOM_SEED, "Random seed",
          cxxopts::value<std::size_t>()->default_value("42"))
        
        (opts::THREAD_BLOCK_SIZE, "Thread block size",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_DIMS_X, "Warp dimension X",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_DIMS_Y, "Warp dimension Y",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_TILE_DIMS_X, "Warp tile dimension X",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::WARP_TILE_DIMS_Y, "Warp tile dimension Y",
          cxxopts::value<std::size_t>()->default_value("0"))
        
        (opts::STREAMING_DIRECTION, "Streaming direction (in-x|in-y|naive)",
          cxxopts::value<std::string>()->default_value("naive"))

        (opts::STATE_BITS_COUNT, "Number of state bits",
          cxxopts::value<std::size_t>()->default_value("64"))

        (opts::BASE_GRID_ENCODING, "Base grid encoding (char|int)",
          cxxopts::value<std::string>()->default_value("char"))

        (opts::TAG, "Tag", cxxopts::value<std::string>()->default_value(""))

        (opts::COLLECT_TOUCHED_TILES_STATS, "Collect touched tiles stats",
          cxxopts::value<bool>()->default_value("false"))
          
        (opts::TEMPORAL_STEPS, "Number of temporal steps",
          cxxopts::value<std::size_t>()->default_value("1"));
      ;

      auto result = optConfig.parse(argc, argv);

      if (result.count("help")) {
        std::cout << optConfig.help() << std::endl;
        exit(0);
      }

      ExperimentParams params;
      params.algorithm_name = result[opts::ALGORITHM_NAME].as<std::string>();
      std::size_t gx = result[opts::GRID_DIMENSIONS_X].as<std::size_t>();
      std::size_t gy = result[opts::GRID_DIMENSIONS_Y].as<std::size_t>();

      params.grid_dimensions = {gx, gy};
      params.iterations = result[opts::ITERATIONS].as<std::size_t>();
      
      params.max_runtime_seconds = result[opts::MAX_RUNTIME_SECONDS].as<std::size_t>();

      params.warmup_rounds = result[opts::WARMUP_ROUNDS].as<std::size_t>();
      params.measurement_rounds = result[opts::MEASUREMENT_ROUNDS].as<std::size_t>();

      params.data_loader_name = result[opts::DATA_LOADER_NAME].as<std::string>();
      params.pattern_expression = result[opts::PATTERN_EXPRESSION].as<std::string>();

      params.measure_speedup = result[opts::MEASURE_SPEEDUP].as<bool>();
      params.speedup_bench_algorithm_name =
        result[opts::SPEEDUP_BENCH_ALGORITHM_NAME].as<std::string>();

      params.validate = result[opts::VALIDATE].as<bool>();
      params.print_validation_diff = result[opts::PRINT_VALIDATION_DIFF].as<bool>();
      params.use_contracted_diff = result[opts::USE_CONTRACTED_DIFF].as<bool>();
      params.validation_algorithm_name =
        result[opts::VALIDATION_ALGORITHM_NAME].as<std::string>();

      params.animate_output = result[opts::ANIMATE_OUTPUT].as<bool>();
      params.colorful = result[opts::COLORFUL].as<bool>();
      params.random_seed = result[opts::RANDOM_SEED].as<std::size_t>();
      params.state_bits_count = result[opts::STATE_BITS_COUNT].as<std::size_t>();

      params.thread_block_size = result[opts::THREAD_BLOCK_SIZE].as<std::size_t>();

      params.warp_dims_x = result[opts::WARP_DIMS_X].as<std::size_t>();
      params.warp_dims_y = result[opts::WARP_DIMS_Y].as<std::size_t>();

      params.warp_tile_dims_x = result[opts::WARP_TILE_DIMS_X].as<std::size_t>();
      params.warp_tile_dims_y = result[opts::WARP_TILE_DIMS_Y].as<std::size_t>();

      auto dir_str = result[opts::STREAMING_DIRECTION].as<std::string>();

      if (dir_str == "in-x")       { params.streaming_direction = StreamingDirection::in_X; }
      else if (dir_str == "in-y")  { params.streaming_direction = StreamingDirection::in_Y; }
      else if (dir_str == "naive") { params.streaming_direction = StreamingDirection::NAIVE; }
      else                         { throw std::runtime_error("Invalid streaming direction"); }

      params.colorful = result[opts::COLORFUL].as<bool>();

      params.state_bits_count = result[opts::STATE_BITS_COUNT].as<std::size_t>();

      params.base_grid_encoding = result[opts::BASE_GRID_ENCODING].as<std::string>();

      params.tag = result[opts::TAG].as<std::string>();

      params.collect_touched_tiles_stats = result[opts::COLLECT_TOUCHED_TILES_STATS].as<bool>();
      
      params.temporal_steps = result[opts::TEMPORAL_STEPS].as<std::size_t>();

      return params;
    }

};

} // namespace infrastructure

#endif // INFRASTRUCTURE_EXPERIMENT_PARAMS_HPP
