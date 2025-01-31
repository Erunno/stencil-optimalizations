# Program Argument Documentation  

The following table documents the various program arguments.

## Index  

| Argument | Type | Description |
| -------- | ---- | ----------- |
| `--algorithm` | `string` | The algorithm to be tested. |
| `--grid-dimensions-x` | `int` | Number of columns in the simulation grid. |
| `--grid-dimensions-y` | `int` | Number of rows in the simulation grid. |
| `--iterations` | `int` | Number of simulation iterations to run. |
| `--max-runtime-seconds` | `float` | Maximum execution time before stopping the simulation. |
| `--base-grid-encoding` | ( `"char"` \| `"int"` ) | Encoding used by the baseline implementation. |
| `--warmup-rounds` | `int` | Number of warm-up iterations before measurements start. |
| `--measurement-rounds` | `int` | Number of iterations used for performance measurement. |
| `--data-loader` | `string` | Method for initializing the grid (see details below). |
| `--pattern-expression` | `string` | Expression defining an initial pattern in the grid. |
| `--measure-speedup` | `bool` | Whether to compute speedup relative to a baseline. |
| `--speedup-bench-algorithm` | `string` | Algorithm to use as a baseline for speedup measurement. |
| `--validate` | `bool` | Whether to validate the output against a reference implementation. |
| `--validation-algorithm` | `string` | Algorithm used for result validation. |
| `--colorful` | `bool` | Enables or disables colored console output. |
| `--random-seed` | `int` | Seed for random number generation (ensures reproducibility). |
| `--thread-block-size` | `int` | CUDA thread block size (only applicable for GPU runs). |
| `--state-bits-count` | ( `32` \| `64` ) | Number of state bits used per block in the case "Reduced work" algorithm. |
| `--tag` | `string` | Optional label for tagging and organizing experiment results. |

## Algorithm Keys  

Each implemented algorithm has a unique key. The following table provides an overview of the available algorithms and their corresponding keys:

| Algorithm                       | Algorithm Key                        | Description |
|----------------------------------|--------------------------------------|-------------|
| **Baselines**                    | `gol-cuda-naive`                     | A straightforward CUDA implementation of Game of Life. |
|                                  | `eff-baseline`                       | Baseline implementation from external work for comparison. |
|                                  | `eff-baseline-shm`                   | Shared memory-optimized baseline implementation. |
| **Linear Bitwise Optimization**  | `gol-cuda-naive-bitwise-cols-32`     | 32-bit column-wise bitwise encoding. |
|                                  | `gol-cuda-naive-bitwise-cols-64`     | 64-bit column-wise bitwise encoding. |
| **Tiled Bitwise Optimization**   | `gol-cuda-naive-bitwise-tiles-32`    | 32-bit tile-based bitwise encoding. |
|                                  | `gol-cuda-naive-bitwise-tiles-64`    | 64-bit tile-based bitwise encoding. |
| **Work Reduction Optimization**  | `gol-cuda-local-one-cell-cols-32`    | 32-bit column-wise bitwise encoding. |
|                                  | `gol-cuda-local-one-cell-cols-64`    | 64-bit column-wise bitwise encoding. |
|                                  | `gol-cuda-local-one-cell-32--bit-tiles` | 32-bit tile-based bitwise encoding. |
|                                  | `gol-cuda-local-one-cell-64--bit-tiles` | 64-bit tile-based bitwise encoding. |
| **Packed**                       | `eff-sota-packed-32`                 | State-of-the-art packed encoding with 32-bit words. |
|                                  | `eff-sota-packed-64`                 | State-of-the-art packed encoding with 64-bit words. |

*Note: If any of the keys are not working, please refer to the actual implementation. Keys are defined [here](./src/infrastructure/experiment_manager.hpp#L60) and [here](./src/algorithms/rel-work/eff-sim-ex-of-cell-auto-GPU/register-all-algs.hpp#L20).*

### CPU Algorithm Keys  

The following table lists the available CPU-based Game of Life algorithms and their corresponding keys:

| Algorithm                        | Algorithm Key                                                              | Description |
|-----------------------------------|----------------------------------------------------------------------------|-------------|
| **Baseline**                     | `gol-cpu-naive`                                                           | Basic CPU implementation of Game of Life with no optimizations. |
| **Naive Linear Bitwise**          | `gol-cpu-bitwise-cols-naive-16`, `gol-cpu-bitwise-cols-naive-32`, `gol-cpu-bitwise-cols-naive-64` | Naive implementation using column-wise bitwise encoding with 16, 32, or 64 bits. |
| **Naive Tiled Bitwise**           | `gol-cpu-bitwise-tiles-naive-16`, `gol-cpu-bitwise-tiles-naive-32`, `gol-cpu-bitwise-tiles-naive-64` | Naive implementation using tile-based bitwise encoding with 16, 32, or 64 bits. |
| **Linear Bitwise (Template)**     | `gol-cpu-bitwise-cols-16`, `gol-cpu-bitwise-cols-32`, `gol-cpu-bitwise-cols-64` | Optimized column-wise bitwise encoding using templates with 16, 32, or 64 bits. |
| **Linear Bitwise (Generated Macros)** | `gol-cpu-bitwise-cols-macro-16`, `gol-cpu-bitwise-cols-macro-32`, `gol-cpu-bitwise-cols-macro-64` | Optimized column-wise bitwise encoding with generated macros for 16, 32, or 64 bits. |
| **Tiled Bitwise (Generated Macros)** | `gol-cpu-bitwise-tiles-macro-16`, `gol-cpu-bitwise-tiles-macro-32`, `gol-cpu-bitwise-tiles-macro-64` | Optimized tile-based bitwise encoding with generated macros for 16, 32, or 64 bits. |

## Data loaders

The data loaders are responsible for initializing the grid with specific patterns. Below are the possible values for the `--data-loader` and `--pattern-expression` options:

### `--data-loader` Options

- **random-ones-zeros**: Initializes the grid with random ones and zeros.
- **always-changing**: Fills the space with blinkers, ensuring that the entire grid constantly changes.
- **zeros**: Initializes the grid with all zeros.
- **lexicon**: Uses a predefined lexicon pattern for initialization (see `--pattern-expression`).

### `--pattern-expression`

This option defines the specific patterns to load into the grid. Patterns can be specified as shown below:

- **blinker[10,10]**: A blinker pattern placed at position (10, 10).
- **glider[3,3]**: A glider pattern starting at position (3, 3).
- **spacefiller[400,600]**: A spacefiller pattern placed at coordinates (400, 600).
- **gosper-glider-gun[0,0]**: A Gosper Glider Gun pattern at position (0, 0).
- There are many other possible patterns available from the lexicon, which can be found in the [lexicon](./src/infrastructure/gol-lexicon/lexicon.txt).

Multiple patterns can be combined in a single expression, separated by spaces. For example:

- `--pattern-expression="glider[3,3] glider[10,10] glider[20,20]"` would place three gliders at different positions.
