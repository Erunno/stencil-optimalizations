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

__base_dim_size=$((64 * 3 * 5 * 7)) # 6720
__16k_like=$(($__base_dim_size * 3)) # 20160 

# ALGORITHM="eff-baseline"
# ALGORITHM="eff-baseline-shm"
# ALGORITHM="eff-baseline-texture"
# ALGORITHM="eff-sota-packed-32"
# ALGORITHM="eff-sota-packed-64"

# ALGORITHM="gol-fujita-64"

# ALGORITHM="gol-cpu-bitwise-wrows-naive-64"
# ALGORITHM="gol-cpu-bitwise-wrows-simd-64"
# ALGORITHM="gol-cuda-naive-bitwise-wrows-64"
# ALGORITHM="gol-cuda-naive-just-tiling-64--wrows"

# ALGORITHM="gol-cuda-naive-bitwise-half-packed-rows-64"
# ALGORITHM="gol-cuda-naive-bitwise-fully-packed-rows-64"

# ALGORITHM="gol-cuda-naive-bitwise-adder-64"

# ALGORITHM="gol-cuda-temporal-simple-64"
# ALGORITHM="gol-cuda-temporal-simple-full-adder-64"

# ALGORITHM="gol-cuda-naive-full-adder-tiles-64"
# ALGORITHM="gol-cuda-temporal-simple-full-adder-64"
ALGORITHM="gol-cuda-simple-warp-ex-full-adder-rows-64"

# ALGORITHM="gol-cpu-naive"
# ALGORITHM="gol-cpu-bitwise-tiles-macro-64"
# ALGORITHM="gol-cpu-bitwise-cols-naive-32"
# ALGORITHM="gol-cpu-bitwise-tiles-naive-64"

# ALGORITHM="gol-cpu-bitwise-cols-macro-64"
# ALGORITHM="gol-cpu-bitwise-cols-naive-64"
# ALGORITHM="gol-cuda-naive-bitwise-tiles-32"
# ALGORITHM="gol-cuda-naive-bitwise-tiles-64"
# ALGORITHM="gol-cuda-naive-bitwise-no-macro-64"
# ALGORITHM="an5d-cpu-64"
# ALGORITHM="cuda-memcpy"
# ALGORITHM="gol-cuda-naive"
# ALGORITHM="gol-cuda-naive-bitwise-cols-32"
# ALGORITHM="gol-cuda-naive-bitwise-cols-64"
# ALGORITHM="gol-cuda-naive-local-64--bit-tiles"
# ALGORITHM="gol-cuda-naive-local-64"
# ALGORITHM="gol-cuda-naive-just-tiling-cols-64"
# ALGORITHM="gol-cuda-naive-just-tiling-64--bit-tiles"
# ALGORITHM="gol-cuda-naive-just-tiling-32--bit-tiles"
# ALGORITHM="gol-cuda-naive-just-tiling-cols-64"
# ALGORITHM="gol-cuda-local-one-cell-32--bit-tiles"
# ALGORITHM="gol-cuda-local-one-cell-64--bit-tiles"
# GRID_DIMENSIONS_X=$(($__6 * 30))
# GRID_DIMENSIONS_Y=$__6
# GRID_DIMENSIONS_X=$((8 * 6))
# GRID_DIMENSIONS_Y=$((8 * 6))
GRID_DIMENSIONS_X=$(($__16k_like * 2))
GRID_DIMENSIONS_Y=$__16k_like
ITERATIONS=$((10 * 1024))
# ITERATIONS="1"

BASE_GRID_ENCODING="char"
# BASE_GRID_ENCODING="int"

WARMUP_ROUNDS="2"
MEASUREMENT_ROUNDS="1"

DATA_LOADER_NAME="random-ones-zeros"
# DATA_LOADER_NAME="always-changing"
# DATA_LOADER_NAME="zeros"
DATA_LOADER_NAME="lexicon"
# PATTERN_EXPRESSION="blinker[10,10]"
# PATTERN_EXPRESSION="glider[3,3] glider[10,10] glider[20,20]"
PATTERN_EXPRESSION="spacefiller[$((GRID_DIMENSIONS_X/2)),$((GRID_DIMENSIONS_Y/2))]"
# PATTERN_EXPRESSION="gosper-glider-gun[0,0]"

# 6x5 sp & 16k & 10000 iters --> total cca 63w/37off workload (on 64 bit)
# PATTERN_EXPRESSION="spacefiller[2340, 2730]; spacefiller[2340, 5460]; spacefiller[2340, 8190]; spacefiller[2340, 10920]; spacefiller[2340, 13650]; spacefiller[4680, 2730]; spacefiller[4680, 5460]; spacefiller[4680, 8190]; spacefiller[4680, 10920]; spacefiller[4680, 13650]; spacefiller[7020, 2730]; spacefiller[7020, 5460]; spacefiller[7020, 8190]; spacefiller[7020, 10920]; spacefiller[7020, 13650]; spacefiller[9360, 2730]; spacefiller[9360, 5460]; spacefiller[9360, 8190]; spacefiller[9360, 10920]; spacefiller[9360, 13650]; spacefiller[11700, 2730]; spacefiller[11700, 5460]; spacefiller[11700, 8190]; spacefiller[11700, 10920]; spacefiller[11700, 13650]; spacefiller[14040, 2730]; spacefiller[14040, 5460]; spacefiller[14040, 8190]; spacefiller[14040, 10920]; spacefiller[14040, 13650];"
# 5x4 sp & 16k & 10000 iters --> total cca 53w/47off workload (on 64 bit)
# PATTERN_EXPRESSION="spacefiller[2730, 3276]; spacefiller[2730, 6552]; spacefiller[2730, 9828]; spacefiller[2730, 13104]; spacefiller[5460, 3276]; spacefiller[5460, 6552]; spacefiller[5460, 9828]; spacefiller[5460, 13104]; spacefiller[8190, 3276]; spacefiller[8190, 6552]; spacefiller[8190, 9828]; spacefiller[8190, 13104]; spacefiller[10920, 3276]; spacefiller[10920, 6552]; spacefiller[10920, 9828]; spacefiller[10920, 13104]; spacefiller[13650, 3276]; spacefiller[13650, 6552]; spacefiller[13650, 9828]; spacefiller[13650, 13104];"
# 4x4 sp & 16k & 10000 iters
# PATTERN_EXPRESSION="spacefiller[5461, 5461]; spacefiller[5461, 10922]; spacefiller[10922, 5461]; spacefiller[10922, 10922];"
# 4x4 sp & 16k & 10000 iters --> total cca 50w/50off workload (on 64 bit) & 
# PATTERN_EXPRESSION="spacefiller[3276, 3276]; spacefiller[3276, 6552]; spacefiller[3276, 9828]; spacefiller[3276, 13104]; spacefiller[6552, 3276]; spacefiller[6552, 6552]; spacefiller[6552, 9828]; spacefiller[6552, 13104]; spacefiller[9828, 3276]; spacefiller[9828, 6552]; spacefiller[9828, 9828]; spacefiller[9828, 13104]; spacefiller[13104, 3276]; spacefiller[13104, 6552]; spacefiller[13104, 9828]; spacefiller[13104, 13104];"
# 4x3 sp & 16k & 10000 iters --> total cca 42w/58off workload (on 64 bit)
# PATTERN_EXPRESSION="spacefiller[3276, 4096]; spacefiller[3276, 8192]; spacefiller[3276, 12288]; spacefiller[6552, 4096]; spacefiller[6552, 8192]; spacefiller[6552, 12288]; spacefiller[9828, 4096]; spacefiller[9828, 8192]; spacefiller[9828, 12288]; spacefiller[13104, 4096]; spacefiller[13104, 8192]; spacefiller[13104, 12288];"
# 3x3 sp & 16k & 10000 iters --> total cca 33w/67off workload (on 64 bit)
# PATTERN_EXPRESSION="spacefiller[4096, 4096]; spacefiller[4096, 8192]; spacefiller[4096, 12288]; spacefiller[8192, 4096]; spacefiller[8192, 8192]; spacefiller[8192, 12288]; spacefiller[12288, 4096]; spacefiller[12288, 8192]; spacefiller[12288, 12288];"


MEASURE_SPEEDUP="true"
# MEASURE_SPEEDUP="false"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-bitwise-tiles-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-just-tiling-64--bit-tiles"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-naive"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cpu-bitwise-cols-naive-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-bitwise-cols-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive"
# SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive-just-tiling-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="eff-sota-packed-64"
# SPEEDUP_BENCH_ALGORITHM_NAME="eff-sota-packed-32"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-fujita-64"

VALIDATE="true"
# VALIDATE="false"
PRINT_VALIDATION_DIFF="false"
# PRINT_VALIDATION_DIFF="true"
# VALIDATION_ALGORITHM_NAME="gol-cpu-naive"
# VALIDATION_ALGORITHM_NAME="gol-cuda-naive"
VALIDATION_ALGORITHM_NAME="gol-fujita-64"

ANIMATE_OUTPUT="false"
# ANIMATE_OUTPUT="true"
COLORFUL="true"

RANDOM_SEED="42"

STATE_BITS_COUNT="64"
# STATE_BITS_COUNT="32"

THREAD_BLOCK_SIZE="1024"
# THREAD_BLOCK_SIZE="512"
# THREAD_BLOCK_SIZE="256"
# THREAD_BLOCK_SIZE="128"
# THREAD_BLOCK_SIZE="64"

TEMPORAL_STEPS="8"
# TEMPORAL_STEPS="4"
# TEMPORAL_STEPS="2"

WARP_DIMS_X="32"
WARP_DIMS_Y="1"

WARP_TILE_DIMS_X="32"
WARP_TILE_DIMS_Y="1"

STREAMING_DIRECTION="in-x"
# STREAMING_DIRECTION="in-y"
# STREAMING_DIRECTION="naive"
MAX_RUNTIME_SECONDS="10000"

TAG="test-run"

# COLLECT_TOUCHED_TILES_STATS="true"
COLLECT_TOUCHED_TILES_STATS="false"

# srun -p gpu-short -A kdss --cpus-per-task=64 --mem=256GB --gres=gpu:L40 --time=2:00:00 $GOL_EXE_NAME \
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
    --temporal-steps="$TEMPORAL_STEPS" \
    --tag="$TAG" \
    --collect-touched-tiles-stats="$COLLECT_TOUCHED_TILES_STATS" \
