#ifndef eff_sim_REGISTER_ALL_ALGS_HPP
#define eff_sim_REGISTER_ALL_ALGS_HPP

#include "../../../infrastructure/algorithm.hpp"
#include "../../../infrastructure/algorithm_repository.hpp"

#include "baseline/GOL.hpp"
#include "packed/GOL.hpp"

namespace algorithms::rel_work {

using namespace eff_sim_ex_of_cell_auto_GPU;

struct Efficient_simulation_execution_of_cellular_automata_on_GPU {

    template <typename grid_cell_t>
    using Repo = infrastructure::AlgorithmRepository<2, grid_cell_t>;

    template <typename grid_cell_t>
    static void register_all_algs(Repo<grid_cell_t>& repo) {

        repo.template register_algorithm<GOL_Baseline<grid_cell_t, BaselineVariant::Basic>>("eff-baseline");
        repo.template register_algorithm<GOL_Baseline<grid_cell_t, BaselineVariant::SharedMemory>>("eff-baseline-shm");
        repo.template register_algorithm<GOL_Baseline<grid_cell_t, BaselineVariant::TextureMemory>>("eff-baseline-texture");

        repo.template register_algorithm<GOL_Packed_sota<grid_cell_t, _32_bit_policy>>("eff-sota-packed-32");
        repo.template register_algorithm<GOL_Packed_sota<grid_cell_t, _64_bit_policy>>("eff-sota-packed-64");
    }
};

}

#endif // eff_sim_REGISTER_ALL_ALGS_HPP
