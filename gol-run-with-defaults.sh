#!/bin/bash

WORK_DIR=$1
cd $WORK_DIR || exit 1

GOL_EXE_NAME="./stencils"

__0="1"
__1="2"
__2="4"
__3="8"
__4="16"
__5="32"
__6="64"
__7="128"
__8="256"
__9="512"
__10="1024"
__11="2048"
__12="4096"
__13="8192"
__14="16384"
__15="32768"
__16="65536"
__17="131072"
__18="262144"
__19="524288"
__20="1048576"
__21="2097152"
__22="4194304"
__23="8388608"
__24="16777216"
__25="33554432"
__26="67108864"
__27="134217728"
__28="268435456"
__29="536870912"
__30="1073741824"
__31="2147483648"

# ALGORITHM="eff-baseline"
# ALGORITHM="eff-baseline-shm"
# ALGORITHM="eff-baseline-texture"
# ALGORITHM="eff-sota-packed-32"
# ALGORITHM="eff-sota-packed-64"
# ALGORITHM="eff-vectors-packed-32"
# ALGORITHM="eff-vectors-packed-64"

# ALGORITHM="gol-cpu-bitwise-wrows-naive-64"
# ALGORITHM="gol-cpu-bitwise-wrows-simd-64"
ALGORITHM="gol-cuda-naive-bitwise-wrows-64"

# ALGORITHM="gol-cpu-bitwise-tiles-macro-64"
# ALGORITHM="gol-cuda-naive-bitwise-tiles-64"
# ALGORITHM="gol-cpu-naive"
# ALGORITHM="gol-cpu-bitwise-cols-naive-64"
# ALGORITHM="gol-cuda-naive-bitwise-tiles-32"
# ALGORITHM="gol-cuda-naive-bitwise-tiles-64"
# ALGORITHM="gol-cpu-bitwise-tiles-naive-64"
# ALGORITHM="gol-cpu-bitwise-cols-naive-32"
# ALGORITHM="gol-cpu-bitwise-cols-macro-64"
# ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"
# ALGORITHM="an5d-cpu-64"
# ALGORITHM="cuda-memcpy"
# ALGORITHM="gol-cuda-naive"
# ALGORITHM="gol-cuda-naive-bitwise-cols-64"
# ALGORITHM="gol-cuda-naive-local-64--bit-tiles"
# ALGORITHM="gol-cuda-naive-local-64"
# ALGORITHM="gol-cuda-naive-just-tiling-cols-64"
# ALGORITHM="gol-cuda-naive-just-tiling-64--bit-tiles"
# ALGORITHM="gol-cuda-naive-just-tiling-cols-64"
GRID_DIMENSIONS_X=$__14
GRID_DIMENSIONS_Y=$__14
# GRID_DIMENSIONS_X=$((8 * 6))
# GRID_DIMENSIONS_Y=$((8 * 6))
ITERATIONS="1024"

BASE_GRID_ENCODING="char"
# BASE_GRID_ENCODING="int"

WARMUP_ROUNDS="0"
MEASUREMENT_ROUNDS="1"

# DATA_LOADER_NAME="random-ones-zeros"
# DATA_LOADER_NAME="always-changing"
# DATA_LOADER_NAME="zeros"
DATA_LOADER_NAME="lexicon"
# PATTERN_EXPRESSION="blinker[10,10]"
# PATTERN_EXPRESSION="glider[3,3] glider[10,10] glider[20,20]"
PATTERN_EXPRESSION="spacefiller[$((GRID_DIMENSIONS_X/2)),$((GRID_DIMENSIONS_Y/2))]"
# PATTERN_EXPRESSION="gosper-glider-gun[0,0]"

MEASURE_SPEEDUP="true"
# MEASURE_SPEEDUP="false"
# SPEEDUP_BENCH_ALGORITHM_NAME="cuda-memcpy"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-naive"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-bitwise-cols-naive-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-bitwise-cols-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-bitwise-tiles-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-just-tiling-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="eff-sota-packed-64"

VALIDATE="true"
# VALIDATE="false"
# PRINT_VALIDATION_DIFF="true"
PRINT_VALIDATION_DIFF="false"
VALIDATION_ALGORITHM_NAME="gol-cpu-naive"
VALIDATION_ALGORITHM_NAME="gol-cuda-naive"

ANIMATE_OUTPUT="false"
# ANIMATE_OUTPUT="true"
COLORFUL="true"

RANDOM_SEED="42"

# STATE_BITS_COUNT="64"
STATE_BITS_COUNT="32"

# THREAD_BLOCK_SIZE="1024"
# THREAD_BLOCK_SIZE="512"
THREAD_BLOCK_SIZE="256"
# THREAD_BLOCK_SIZE="128"
# THREAD_BLOCK_SIZE="64"

WARP_DIMS_X="32"
WARP_DIMS_Y="1"

WARP_TILE_DIMS_X="32"
WARP_TILE_DIMS_Y="8"

STREAMING_DIRECTION="in-x"
# STREAMING_DIRECTION="in-y"
# STREAMING_DIRECTION="naive"
MAX_RUNTIME_SECONDS="10"

TAG="test-run"

# srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:V100 --time=2:00:00 $GOL_EXE_NAME \
srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:H100 --time=2:00:00 $GOL_EXE_NAME \
    --algorithm="$ALGORITHM" \
    --grid-dimensions-x="$GRID_DIMENSIONS_X" \
    --grid-dimensions-y="$GRID_DIMENSIONS_Y" \
    --iterations="$ITERATIONS" \
    --max-runtime-seconds="$MAX_RUNTIME_SECONDS" \
    --base-grid-encoding="$BASE_GRID_ENCODING" \
    --warmup-rounds="$WARMUP_ROUNDS" \
    --measurement-rounds="$MEASUREMENT_ROUNDS" \
    --data-loader="$DATA_LOADER_NAME" \
    --pattern-expression="$PATTERN_EXPRESSION" \
    --measure-speedup="$MEASURE_SPEEDUP" \
    --speedup-bench-algorithm="$SPEEDUP_BENCH_ALGORITHM_NAME" \
    --validate="$VALIDATE" \
    --print-validation-diff="$PRINT_VALIDATION_DIFF" \
    --validation-algorithm="$VALIDATION_ALGORITHM_NAME" \
    --animate-output="$ANIMATE_OUTPUT" \
    --colorful="$COLORFUL" \
    --random-seed="$RANDOM_SEED" \
    --thread-block-size="$THREAD_BLOCK_SIZE" \
    --warp-dims-x="$WARP_DIMS_X" \
    --warp-dims-y="$WARP_DIMS_Y" \
    --warp-tile-dims-x="$WARP_TILE_DIMS_X" \
    --warp-tile-dims-y="$WARP_TILE_DIMS_Y" \
    --streaming-direction="$STREAMING_DIRECTION" \
    --state-bits-count="$STATE_BITS_COUNT" \
    --tag="$TAG" \
