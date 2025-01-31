#!/bin/bash

WORK_DIR=$1
cd $WORK_DIR || exit 1

GOL_EXE_NAME="./stencils"

pow2() {
    echo $((2 ** $1))
}


# ALGORITHM="gol-cuda-naive-bitwise-tiles-32"
ALGORITHM="gol-cuda-naive-bitwise-tiles-64"

# ALGORITHM="gol-cuda-naive"
# ALGORITHM="gol-cuda-naive-bitwise-cols-32"
# ALGORITHM="gol-cuda-naive-bitwise-cols-64"

# ALGORITHM="gol-cuda-local-one-cell-32--bit-tiles"
# ALGORITHM="gol-cuda-local-one-cell-64--bit-tiles"
# ALGORITHM="gol-cuda-local-one-cell-cols-32"
# ALGORITHM="gol-cuda-local-one-cell-cols-64"

# ALGORITHM="eff-baseline"
# ALGORITHM="eff-baseline-shm"
# ALGORITHM="eff-sota-packed-32"
# ALGORITHM="eff-sota-packed-64"

GRID_DIMENSIONS_X=$(pow2 13)
GRID_DIMENSIONS_Y=$(pow2 13)

ITERATIONS="10000"

BASE_GRID_ENCODING="char"
# BASE_GRID_ENCODING="int"

WARMUP_ROUNDS="1"
MEASUREMENT_ROUNDS="1"

# DATA_LOADER_NAME="random-ones-zeros"
# DATA_LOADER_NAME="always-changing"
# DATA_LOADER_NAME="zeros"

DATA_LOADER_NAME="lexicon"
PATTERN_EXPRESSION="blinker[10,10]"
# PATTERN_EXPRESSION="glider[3,3] glider[10,10] glider[20,20]"
# PATTERN_EXPRESSION="spacefiller[$((GRID_DIMENSIONS_X/2)),$((GRID_DIMENSIONS_Y/2))]"
# PATTERN_EXPRESSION="gosper-glider-gun[0,0]"

# used in the paper as "66% busy" - 6x5 spacefiller & 16k grid & 10000 iters
# PATTERN_EXPRESSION="spacefiller[2340, 2730]; spacefiller[2340, 5460]; spacefiller[2340, 8190]; spacefiller[2340, 10920]; spacefiller[2340, 13650]; spacefiller[4680, 2730]; spacefiller[4680, 5460]; spacefiller[4680, 8190]; spacefiller[4680, 10920]; spacefiller[4680, 13650]; spacefiller[7020, 2730]; spacefiller[7020, 5460]; spacefiller[7020, 8190]; spacefiller[7020, 10920]; spacefiller[7020, 13650]; spacefiller[9360, 2730]; spacefiller[9360, 5460]; spacefiller[9360, 8190]; spacefiller[9360, 10920]; spacefiller[9360, 13650]; spacefiller[11700, 2730]; spacefiller[11700, 5460]; spacefiller[11700, 8190]; spacefiller[11700, 10920]; spacefiller[11700, 13650]; spacefiller[14040, 2730]; spacefiller[14040, 5460]; spacefiller[14040, 8190]; spacefiller[14040, 10920]; spacefiller[14040, 13650];"

# used in the paper as "33% busy" - 3x3 sp & 16kx16k grid & 10000 iterations
# PATTERN_EXPRESSION="spacefiller[4096, 4096]; spacefiller[4096, 8192]; spacefiller[4096, 12288]; spacefiller[8192, 4096]; spacefiller[8192, 8192]; spacefiller[8192, 12288]; spacefiller[12288, 4096]; spacefiller[12288, 8192]; spacefiller[12288, 12288];"

MEASURE_SPEEDUP="true"
# MEASURE_SPEEDUP="false"
SPEEDUP_BENCH_ALGORITHM_NAME="gol-cuda-naive"

VALIDATE="true"
# VALIDATE="false"
VALIDATION_ALGORITHM_NAME="gol-cuda-naive"

COLORFUL="true"

RANDOM_SEED="42"

STATE_BITS_COUNT="64"
# STATE_BITS_COUNT="32"

# THREAD_BLOCK_SIZE="1024"
# THREAD_BLOCK_SIZE="512"
THREAD_BLOCK_SIZE="256"
# THREAD_BLOCK_SIZE="128"
# THREAD_BLOCK_SIZE="64"

MAX_RUNTIME_SECONDS="10000"

TAG="test-run"

$GOL_EXE_NAME \
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
    --validation-algorithm="$VALIDATION_ALGORITHM_NAME" \
    --colorful="$COLORFUL" \
    --random-seed="$RANDOM_SEED" \
    --thread-block-size="$THREAD_BLOCK_SIZE" \
    --state-bits-count="$STATE_BITS_COUNT" \
    --tag="$TAG" \
