import sys
from general_job_generation import *

class BenchSetUp:
    SPEED_UP_AND_VALIDATION_OFF = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '
    SPEED_UP_AND_VALIDATION_ON = ' MEASURE_SPEEDUP="false"  VALIDATE="false" '

    GENERAL_SETTINGS = ' ITERATIONS="100000" MAX_RUNTIME_SECONDS="5" RANDOM_SEED="42" WARMUP_ROUNDS="5" MEASUREMENT_ROUNDS="10" COLORFUL="false" ANIMATE_OUTPUT="false" BASE_GRID_ENCODING="char" '

    TEST_CASES = [
        # f' GRID_DIMENSIONS_X="1024"  GRID_DIMENSIONS_Y="1024"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="2048"  GRID_DIMENSIONS_Y="2048"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="4096"  GRID_DIMENSIONS_Y="4096"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="8192"  GRID_DIMENSIONS_Y="8192"  {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        # f' GRID_DIMENSIONS_X="32768" GRID_DIMENSIONS_Y="32768" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
        # f' GRID_DIMENSIONS_X="65536" GRID_DIMENSIONS_Y="65536" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ',
    ]

    DATA_DEPENDANT_CASES = [
        f' GRID_DIMENSIONS_X="16384" GRID_DIMENSIONS_Y="16384" {GENERAL_SETTINGS} {SPEED_UP_AND_VALIDATION_OFF} ITERATIONS="10000" MAX_RUNTIME_SECONDS="5000" ',
    ]

    VARIOUS_DATA_LOADERS = [
        # f' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]" TAG="spacefiller"  ',
        # ' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="gosper-glider-gun[0,0]" TAG="glider-gun"  ',
        ' DATA_LOADER_NAME="zeros" TAG="no-work" ',
        ' DATA_LOADER_NAME="always-changing" TAG="full-work" ',
        ' TAG="66-work" DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[2340, 2730]; spacefiller[2340, 5460]; spacefiller[2340, 8190]; spacefiller[2340, 10920]; spacefiller[2340, 13650]; spacefiller[4680, 2730]; spacefiller[4680, 5460]; spacefiller[4680, 8190]; spacefiller[4680, 10920]; spacefiller[4680, 13650]; spacefiller[7020, 2730]; spacefiller[7020, 5460]; spacefiller[7020, 8190]; spacefiller[7020, 10920]; spacefiller[7020, 13650]; spacefiller[9360, 2730]; spacefiller[9360, 5460]; spacefiller[9360, 8190]; spacefiller[9360, 10920]; spacefiller[9360, 13650]; spacefiller[11700, 2730]; spacefiller[11700, 5460]; spacefiller[11700, 8190]; spacefiller[11700, 10920]; spacefiller[11700, 13650]; spacefiller[14040, 2730]; spacefiller[14040, 5460]; spacefiller[14040, 8190]; spacefiller[14040, 10920]; spacefiller[14040, 13650];" ',
        ' TAG="33-work" DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[4096, 4096]; spacefiller[4096, 8192]; spacefiller[4096, 12288]; spacefiller[8192, 4096]; spacefiller[8192, 8192]; spacefiller[8192, 12288]; spacefiller[12288, 4096]; spacefiller[12288, 8192]; spacefiller[12288, 12288];" ',
    ]
    WHATEVER_DATA_LOADER = [
        f' DATA_LOADER_NAME="lexicon" PATTERN_EXPRESSION="spacefiller[{MID_COORDS_MACRO}]"  ',
    ]

    BEST_CASE_WORK_REDUCTION_LOADER = [
        ' DATA_LOADER_NAME="zeros" TAG="no-work" ',
    ]


simple_cases_base = [
    BenchSetUp.TEST_CASES,
    BenchSetUp.WHATEVER_DATA_LOADER,
]

local_cases_base = [
    BenchSetUp.DATA_DEPENDANT_CASES,
    BenchSetUp.VARIOUS_DATA_LOADERS,
]

local_best_case = [
    BenchSetUp.TEST_CASES,
    BenchSetUp.BEST_CASE_WORK_REDUCTION_LOADER,
]

class HP:
    def __init__(self):
        self.thb_size = None
        self.state_bits_count = None

    def block_size(self, size: int):
        self.thb_size = size
        return self

    def state_bits(self, bits: int):
        self.state_bits_count = bits
        return self
    
    def str(self):
        thb = f' THREAD_BLOCK_SIZE="{self.thb_size}" ' if self.thb_size else ''
        state_bits = f' STATE_BITS_COUNT="{self.state_bits_count}" ' if self.state_bits_count else ''

        return f'{thb}{state_bits}'

per_alg_hps = [
    
    # GPUs
    #   - best hyper-parameters was determined experimentally

    # [['gol-cuda-naive', [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],
    #  [*simple_cases_base, [HP().block_size(128).str()]]],

    # [['gol-cuda-naive-bitwise-cols-32', None],
    #  [*simple_cases_base, [HP().block_size(512).str()]]],
    
    # [['gol-cuda-naive-bitwise-cols-64', None],
    #  [*simple_cases_base, [HP().block_size(512).str()]]],

    # [['gol-cuda-naive-bitwise-tiles-32', None],                                              
    #  [*simple_cases_base, [HP().block_size(128).str()]]],
    
    # [['gol-cuda-naive-bitwise-tiles-64', None],                                              
    #  [*simple_cases_base, [HP().block_size(256).str()]]],

    # [['gol-cuda-naive-bitwise-no-macro-32', None],                                           
    #  [*simple_cases_base, [HP().block_size(128).str()]]],
    
    # [['gol-cuda-naive-bitwise-no-macro-64', None],                                           
    #  [*simple_cases_base, [HP().block_size(256).str()]]],

    # [['gol-cuda-local-one-cell-cols-32', None],                                              
    #  [*local_cases_base, [HP().block_size(256).state_bits(32).str()]],],
    
    # [['gol-cuda-local-one-cell-cols-64', None],                                              
    #  [*local_cases_base, [HP().block_size(128).state_bits(32).str()]]],

    # [['gol-cuda-local-one-cell-32--bit-tiles', None],                                        
    #  [*local_cases_base, [HP().block_size(256).state_bits(32).str()]]],
    
    # [['gol-cuda-local-one-cell-64--bit-tiles', None],                                        
    #  [*local_cases_base, [HP().block_size(128).state_bits(32).str()]]],

    # best of best cases for work reduction
    [['gol-cuda-local-one-cell-64--bit-tiles', None],                                        
     [*local_best_case, [HP().block_size(128).state_bits(32).str()]]],

    # # CPUs

    # [['gol-cpu-naive', [' BASE_GRID_ENCODING="char" ', ' BASE_GRID_ENCODING="int" ']],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-cols-naive-32', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-cols-naive-64', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-tiles-naive-32', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-tiles-naive-64', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-cols-macro-32', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-cols-macro-64', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-tiles-macro-32', None],                                        
    #  [*simple_cases_base]],

    # [['gol-cpu-bitwise-tiles-macro-64', None],                                        
    #  [*simple_cases_base]],

    # Related Work
    
    # [['eff-baseline', None],
    #  [*simple_cases_base, [HP().block_size(1024).str()]]],

    # [['eff-baseline-shm', None],
    #  [*simple_cases_base, [HP().block_size(256).str()]]],

    # # [['eff-baseline-texture', None],
    # #  [*simple_cases_base, [HP().block_size(xx)]]],

    # [['eff-sota-packed-32', None],
    #  [*simple_cases_base, [HP().block_size(1024).str()]]],

    # [['eff-sota-packed-64', None],
    #  [*simple_cases_base, [HP().block_size(1024).str()]]],
]

if len(sys.argv) != 3:
    print('Usage: generate-final-measurements.py <template_name> <workers_count>')
    sys.exit(1)

res = Generator() \
    .set_algs_and_hps(per_alg_hps) \
    .generate_all()

print ('# generated: ', len(res))

template_name = sys.argv[1]
workers_count = int(sys.argv[2])

folder = 'final-measurements'
write_to_files(folder, res, template_name, workers_count)
