#!/bin/bash

echo next-experiment

EXECUTABLE=../build/src/stencils

PATTERN_EXPRESSION=${PATTERN_EXPRESSION:-glider[0,0]}
SPEEDUP_BENCH_ALGORITHM_NAME=${SPEEDUP_BENCH_ALGORITHM_NAME:-gol-cuda-naive}
VALIDATION_ALGORITHM_NAME=${VALIDATION_ALGORITHM_NAME:-gol-cuda-naive}
PRINT_VALIDATION_DIFF=${PRINT_VALIDATION_DIFF:-false}
ANIMATE_OUTPUT=${ANIMATE_OUTPUT:-false}
COLORFUL=${COLORFUL:-false}
RANDOM_SEED=${RANDOM_SEED:-42}
THREAD_BLOCK_SIZE=${THREAD_BLOCK_SIZE:-512}
WARP_DIMS_X=${WARP_DIMS_X:-32}
WARP_DIMS_Y=${WARP_DIMS_Y:-1}
WARP_TILE_DIMS_X=${WARP_TILE_DIMS_X:-32}
WARP_TILE_DIMS_Y=${WARP_TILE_DIMS_Y:-8}
STREAMING_DIRECTION=${STREAMING_DIRECTION:-in-x}
STATE_BITS_COUNT=${STATE_BITS_COUNT:-32}
TAG=${TAG:-""}

$EXECUTABLE \
    --algorithm="$ALGORITHM" \
    --grid-dimensions-x="$GRID_DIMENSIONS_X" \
    --grid-dimensions-y="$GRID_DIMENSIONS_Y" \
    --iterations="$ITERATIONS" \
    --max-runtime-seconds="$MAX_RUNTIME_SECONDS" \
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
    --base-grid-encoding="$BASE_GRID_ENCODING" \
    --tag="$TAG"
