#include "algorithms/_shared/bitwise/general_bit_grid.hpp"
#include "algorithms/_shared/bitwise/bitwise-ops/templated-cols.hpp"
#include "algorithms/_shared/common_grid_types.hpp"
#include "algorithms/_shared/bitwise/bitwise-ops/cpu-ops-interface.hpp"
#include "algorithms/cpu-bitwise-general/gol_cpu_bitwise_general.hpp"
#include "algorithms/cuda-naive-bitwise/gol_cuda_naive_bitwise.hpp"
#include "infrastructure/data_loader.hpp"
#include "infrastructure/experiment_manager.hpp"
#include "infrastructure/experiment_params.hpp"
#include "infrastructure/gol-lexicon/lexicon.hpp"
#include "infrastructure/grid.hpp"
#include <bit>
#include <bitset>
#include <cstdint>

#include "debug_utils/pretty_print.hpp"
#include <iostream>
#include <string>

using namespace debug_utils;

int main(int argc, char** argv) {
    std::cout << "Hello" << std::endl;

    auto params = infrastructure::ParamsParser::parse(argc, argv);

    c::set_colorful(params.colorful);

    std::cout << params.pretty_print() << std::endl;

    if (params.base_grid_encoding == "char") {
        infrastructure::ExperimentManager<common::CHAR> manager;
        manager.run(params);

    } else if (params.base_grid_encoding == "int") {
        infrastructure::ExperimentManager<common::INT> manager;
        manager.run(params);
        
    } else {
        std::cerr << "Invalid base grid encoding: " << params.base_grid_encoding << std::endl;
        exit(1);
    }

    return 0;
}
