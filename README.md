# Slaying a Life 🔥🦠

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENCE) [![doi](https://img.shields.io/badge/DOI-TODO-blue)](todo)

This repository is associated with the following paper:  

```
@article { TODO }
```

![Glider Gun](./glider-gun.gif)

## About  

Iterative stencil loops (ISLs) are widely used in simulations, image processing, and cellular automata. This project focuses on optimizing Conway's Game of Life on GPUs. While existing implementations are efficient, we identified areas for significant improvement. Our approach minimizes unnecessary computations by skipping unchanged grid regions and employs a more efficient bit-wise encoding. The result is a **22× speedup** over a basic GPU implementation, **7.3× faster** performance compared to the packet-coding method, and **18.4× faster** execution than AN5D-generated code.  

## Index  

The paper discusses several optimization techniques. Below is an index to help navigate the repository.  

| Algorithm | Description | Link |  
|-----------|-------------|------|  
| **Baselines** | A straightforward Game of Life implementation, with `int` and `char` variants passed as template parameters. | [Kernel](./src/algorithms/cuda-naive/cuda_naive_kernel.cu#L22) |  
| **Linear Bitwise Optimization** | CUDA kernel for bit-wise encoded approaches. | [Kernel](./src/algorithms/cuda-naive-bitwise/cuda_naive_bitwise_kernel.cu#L33) |  
| | Python script for generating macros. | [Script](./src/algorithms/_shared/bitwise/bitwise-ops/python-macro-generators/cols_macro_gen.py) |  
| | Generated macros. | [Macros](./src/algorithms/_shared/bitwise/bitwise-ops/macro-cols.hpp) |  
| | Templated implementation (on CPUs only) | [Code](./src/algorithms/_shared/bitwise/bitwise-ops/templated-cols.hpp#L31)|
| **Tiled Bitwise Optimization** 🚀 | Uses the same kernel as the Linear approach but modifies `CudaBitwiseOps<word_type, bit_grid_mode>::compute_center_word`, which calls a different macro. | [Kernel](./src/algorithms/cuda-naive-bitwise/cuda_naive_bitwise_kernel.cu#L33) |  
| | Python script for generating macros. | [Script](./src/algorithms/_shared/bitwise/bitwise-ops/python-macro-generators/tiles_macro_gen.py) |  
| | Generated macros. | [Macros](./src/algorithms/_shared/bitwise/bitwise-ops/macro-tiles.hpp) |  
| **Work Reduction Optimization** 🚀 | The main kernel of the algorithm, along with various `__device__` functions used within the kernel. | [Kernel](./src/algorithms/cuda-naive-local-one-cell/cuda_local_one_cell.cu#L173) |  
| **AN5D** | The base directory for AN5D implementations. Each subdirectory contains an implementation for a specific grid size, which must be compiled and run separately. | [AN5D test cases](./AN5D/) |  
| **Packed** | Integrated the Packed algorithm into our framework. *Note: We fixed a minor bug (which did not affect performance) in our codebase.* | [Kernel](./src/algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/packed/GOL_packed.cu#L48) |  
| | Baseline implementations from their work. | [Simple kernel](./src/algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/baseline/GOL_basic.cu#L13), [Shared memory kernel](./src/algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/baseline/GOL_shm.cu#L14), [Texture kernel](./src/algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/baseline/GOL_texture.cu#L13) |  

## Tutorial  

We used `GCC 11.4.1` and `nvcc 12.6.77` to compile the code.  

### Minimal Compilation and Execution  

```bash
$> git clone https://github.com/matyas-brabec/2025-ashes-gpulife
$> cd 2025-ashes-gpulife
$> ./configure.sh
$> ./run.sh
```

Depending on your environment, you may need to modify some variables. The `./run.sh` script compiles the code and then runs it with default parameters. The executable should be located in `./build/src/stencils`. If this is not the case, update the [WORK_DIR](./run.sh#L13) and [GOL_EXE_NAME](./gol-run-with-defaults.sh#L6) variables accordingly.  

You can modify the program's runtime parameters in the [./gol-run-with-defaults.sh](./gol-run-with-defaults.sh) script. A detailed explanation of these parameters can be found in the [program argument documentation](./README-program-arguments.md).  

## Results  

Our measurement results are available in the [./cluster-experiments/final-measurements](./cluster-experiments/final-measurements) directory. It contains subfolders for each tested architecture: `hopper`, `ampere`, and `volta` (further information about our cluster [here](https://gitlab.mff.cuni.cz/mff/hpc/clusters)).

Graphs similar to those in the paper can be found in [./cluster-experiments/generated-graphs](./cluster-experiments/generated-graphs). To generate graphs from your own measurements, use the [result_analysis.py](./cluster-experiments/result_analysis.py) script. You may need to modify the following control variables:  

- [BASE_DIR](./cluster-experiments/result_analysis.py#L11)  
- [ARCHITECTURES](./cluster-experiments/result_analysis.py#L10)
- [MODE](./cluster-experiments/result_analysis.py#L15)

### Replication  

To replicate our experiments exactly, run the script [./cluster-experiments/run-all-experiments.sh](./cluster-experiments/run-all-experiments.sh):  

```bash
$> cd cluster-experiments  # The working directory must be this folder
$> ./run-all-experiments.sh > my-measurements/all-experiments.out
$> python ./result_analysis.py  # You will need to adjust control variables as discussed
```

## Contact us

If you have any questions regarding the optimizations, our implementation, or if you have any suggestions, please feel free to contact us by raising [an issue](https://github.com/matyas-brabec/2025-ashes-gpulife/issues)!

## License

This source code is licensed under the [MIT license](./LICENCE).
