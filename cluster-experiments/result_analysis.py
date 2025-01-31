import os
import random
import results_abstraction as ra
import matplotlib.pyplot as plt
import numpy as np

# USE_DEBUG_NAMES = False
USE_DEBUG_NAMES = False

ARCHITECTURES = ['hopper', 'ampere', 'volta']
BASE_DIR = './final-measurements/{architecture}'

GRAPH_DIR = './generated-graphs'

# MODE='png'
MODE='pdf'

X_LABEL_FONT_SIZE = 20
Y_LABEL_FONT_SIZE = 20
LEGEND_FONT_SIZE = 13
X_TICKS_FONT_SIZE = 17
Y_TICKS_FONT_SIZE = 17

X_TICKS_OFFSET = -0.02
Y_LABEL_OFFSET = 12

class LegendNames:
    @staticmethod
    def get(alg):
        key = '_'.join([k[1] for k in alg])

        if USE_DEBUG_NAMES:
            names = LegendNames.get_debug_names()
        else:
            names = LegendNames.get_paper_names()

        try:
            return names[key]
        except:
            print('No name for:', key, ' --> using debug names')
            return LegendNames.get_debug_names()[key]

    @staticmethod
    def get_debug_names():
        return {
            'gol-cpu-naive_char': 'CPU Naive (char)',
            'gol-cpu-naive_int': 'CPU Naive (int)',
            'gol-cpu-bitwise-cols-naive-32': 'CPU Bitwise Cols Naive 32',
            'gol-cpu-bitwise-cols-naive-64': 'CPU Bitwise Cols Naive 64',
            'gol-cpu-bitwise-cols-macro-32': 'CPU Bitwise Cols Macro 32',
            'gol-cpu-bitwise-cols-macro-64': 'CPU Bitwise Cols Macro 64',
            'gol-cpu-bitwise-tiles-naive-32': 'CPU Bitwise Tiles Naive 32',
            'gol-cpu-bitwise-tiles-naive-64': 'CPU Bitwise Tiles Naive 64',
            'gol-cpu-bitwise-tiles-macro-32': 'CPU Bitwise Tiles Macro 32',
            'gol-cpu-bitwise-tiles-macro-64': 'CPU Bitwise Tiles Macro 64',

            'gol-cuda-naive_char': 'CUDA Naive (char)',
            'gol-cuda-naive_int': 'CUDA Naive (int)',
            'gol-cuda-naive-bitwise-no-macro-32': 'CUDA Bitwise Naive 32',
            'gol-cuda-naive-bitwise-no-macro-64': 'CUDA Bitwise Naive 64',
            'gol-cuda-naive-bitwise-cols-32': 'CUDA Bitwise Cols Macro 32',
            'gol-cuda-naive-bitwise-cols-64': 'CUDA Bitwise Cols Macro 64',
            'gol-cuda-naive-bitwise-tiles-32': 'CUDA Bitwise Tiles Macro 32',
            'gol-cuda-naive-bitwise-tiles-64': 'CUDA Bitwise Tiles Macro 64',
            'gol-cuda-local-one-cell-cols-32': 'CUDA Local Cols 32',
            'gol-cuda-local-one-cell-cols-64': 'CUDA Local Cols 64',
            'gol-cuda-local-one-cell-32--bit-tiles': 'CUDA Local Bit Tiles 32',
            'gol-cuda-local-one-cell-64--bit-tiles': 'CUDA Local Bit Tiles 64',

            'eff-baseline': 'Eff Baseline',
            'eff-baseline-shm': 'Eff Baseline SHM',
            'eff-sota-packed-32': 'Eff SOTA Packed 32',
            'eff-sota-packed-64': 'Eff SOTA Packed 64',

            'gol-cuda-local-one-cell-64--bit-tiles_no-work': 'CUDA Local Bit Tiles 64 No Work',
            'gol-cuda-local-one-cell-64--bit-tiles_full-work': 'CUDA Local Bit Tiles 64 Full Work',
            'gol-cuda-local-one-cell-64--bit-tiles_glider-gun': 'CUDA Local Bit Tiles 64 Glider Gun',
            'gol-cuda-local-one-cell-64--bit-tiles_spacefiller': 'CUDA Local Bit Tiles 64 Spacefiller',

            'an5d': 'AN5D',

        }


    @staticmethod
    def get_paper_names():
        return {
            'gol-cuda-naive-bitwise-cols-32': 'Core Linear 32-bit',
            'gol-cuda-naive-bitwise-cols-64': 'Core Linear 64-bit',
            'gol-cuda-naive-bitwise-tiles-32': 'Core Tiled 32-bit',
            'gol-cuda-naive-bitwise-tiles-64': 'Core Tiled 64-bit',
            
            'eff-sota-packed-32': 'Packed 32-bit (SOTA)',
            
            'gol-cuda-naive_char': 'Baseline (char)',
            'gol-cuda-naive_int': 'Baseline (int)',
            
            'an5d': 'AN5D',

            'gol-cuda-local-one-cell-cols-64': 'Optimized Linear 64-bit',
            'gol-cuda-local-one-cell-64--bit-tiles': 'Optimized Tiled 64-bit',
            'gol-cuda-local-one-cell-64--bit-tiles_no-work': 'Optimized Tiled 64-bit (Optimal)',
        }

class ALG_LIST:
    cpu_naive_char =                     [(ra.Key.algorithm_name, 'gol-cpu-naive'), (ra.Key.base_grid_encoding, 'char')]
    cpu_naive_int =                      [(ra.Key.algorithm_name, 'gol-cpu-naive'), (ra.Key.base_grid_encoding, 'int')]
    
    cpu_bitwise_cols_naive_32 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-naive-32')]
    cpu_bitwise_cols_naive_64 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-naive-64')]
    
    cpu_bitwise_cols_macro_32 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-macro-32')]
    cpu_bitwise_cols_macro_64 =          [(ra.Key.algorithm_name, 'gol-cpu-bitwise-cols-macro-64')]

    cpu_bitwise_tiles_naive_32 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-32')]    
    cpu_bitwise_tiles_naive_64 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-64')]

    cpu_bitwise_tiles_macro_32 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-32')]
    cpu_bitwise_tiles_macro_64 =         [(ra.Key.algorithm_name, 'gol-cpu-bitwise-tiles-naive-64')]
    
    cuda_naive_char =                    [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'char')]
    cuda_naive_int =                     [(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'int')]

    cuda_naive_bitwise_no_macro_32 =     [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-32')]
    cuda_naive_bitwise_no_macro_64 =     [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-no-macro-64')]
    
    cuda_naive_bitwise_cols_32 =         [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-32')]
    cuda_naive_bitwise_cols_64 =         [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-64')]

    cuda_naive_bitwise_tiles_32 =        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-tiles-32')]
    cuda_naive_bitwise_tiles_64 =        [(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-tiles-64')]
    
    cuda_local_one_cell_cols_32 =        [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-32')]
    cuda_local_one_cell_cols_64 =        [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-64')]

    cuda_local_one_cell_bit_tiles_32 =   [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-32--bit-tiles')]
    cuda_local_one_cell_bit_tiles_64 =   [(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-64--bit-tiles')]

    cuda_an5d =   [(ra.Key.algorithm_name, 'an5d')]

    eff_baseline = [(ra.Key.algorithm_name, 'eff-baseline')]
    eff_baseline_shm = [(ra.Key.algorithm_name, 'eff-baseline-shm')]
    
    eff_sota_packed_32 = [(ra.Key.algorithm_name, 'eff-sota-packed-32')]
    eff_sota_packed_64 = [(ra.Key.algorithm_name, 'eff-sota-packed-64')]

    ALGS = [
        cpu_naive_char,
        cpu_naive_int,
        cpu_bitwise_cols_naive_32,
        cpu_bitwise_cols_naive_64,
        cpu_bitwise_cols_macro_32,
        cpu_bitwise_cols_macro_64,
        cuda_naive_char,
        cuda_naive_int,
        cuda_naive_bitwise_no_macro_32,
        cuda_naive_bitwise_no_macro_64,
        cuda_naive_bitwise_cols_32,
        cuda_naive_bitwise_cols_64,
    ]

    g_1024 = [(ra.Key.grid_dimensions, '1024x1024')]
    g_2048 = [(ra.Key.grid_dimensions, '2048x2048')]
    g_4096 = [(ra.Key.grid_dimensions, '4096x4096')]
    g_8192 = [(ra.Key.grid_dimensions, '8192x8192')]
    g_16384 = [(ra.Key.grid_dimensions, '16384x16384')]
    g_32768 = [(ra.Key.grid_dimensions, '32768x32768')]
    g_65536 = [(ra.Key.grid_dimensions, '65536x65536')]

    data__no_work = [(ra.Key.tag, 'no-work')]
    data__full_work = [(ra.Key.tag, 'full-work')]
    data__glider_gun = [(ra.Key.tag, 'glider-gun')]
    data__spacefiller = [(ra.Key.tag, 'spacefiller')]
    data__33_work = [(ra.Key.tag, '33-work')]
    data__66_work = [(ra.Key.tag, '66-work')]

def from_ms_to_pico_seconds(val):
    return val * 1_000_000_000 if val is not None else None

class TimePerCellPerIter__InputSize:

    def __init__(self, results: ra.Results):
        self.results = results
        self.tested_grids = []
        self.algs = []
        self.PLOT_NAME = 'time_per_cell_per_iter__input_size'
        self.position = None
        self.bbox = None
        self.markers = ['o', 's', 'v', 'x', 'd', 'p', 'h', 'H', 'D', 'P', '*', '+', '|', '_', '1', '2', '3', '4', '8', '<', '>', '^', 'v']
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.linestyles = ['-', '--', '-.', ':']

    def set_algs(self, algs):
        self.algs = algs
        return self

    def set_grids(self, grids):
        self.tested_grids = grids
        return self
    
    def set_position(self, position):
        self.position = position
        return self
    
    def set_bbox(self, bbox):
        self.bbox = bbox
        return self
    
    def set_plot_name(self, name):
        self.PLOT_NAME = name
        return self
    
    def set_markers(self, markers):
        self.markers = markers
        return self
    
    def set_colors(self, colors):
        self.colors = colors
        return self

    def set_linestyles(self, linestyles):
        self.linestyles = linestyles
        return self

    def gen_graphs(self):
        plt.figure(figsize=(8, 6))

        means = []
        
        for alg in self.algs:
            alg_values = []
            alg_box_data = []

            for grid in self.tested_grids:
                exp = self.results.get_experiments_with([*alg, *grid])

                if not exp:
                    alg_values.append(None)
                    alg_box_data.append(None)
                    continue

                if len(exp) != 1:
                    print('Found more than 1 exp:', len(exp), 'for:', alg, grid)

                measurements = exp[0].get_measurement_set()
                
                time = measurements.get_median(lambda m: m.compute_runtime_per_cell_per_iter())

                if time is None:
                    print ('None value for:', alg, grid)
                    time = 0

                alg_values.append(time)
                
            means.append(alg_values)

        x_labels = [grid[0][1] for grid in self.tested_grids]
        x_positions = range(len(x_labels))
        
        for i, mean_vals in enumerate(means):
            mean_vals = [from_ms_to_pico_seconds(v) for v in mean_vals]
            plt.plot(
                x_positions,
                mean_vals,
                label=str(self.algs[i]),
                color=self.colors[i % len(self.colors)],
                marker=self.markers[i % len(self.markers)],
                linestyle=self.linestyles[i % len(self.linestyles)]
            )

        plt.xticks(x_positions, x_labels, rotation=0, fontsize=X_TICKS_FONT_SIZE, y=X_TICKS_OFFSET)

        # plt.xlabel("Grid Size", fontsize=X_LABEL_FONT_SIZE)
        plt.ylabel("Time per one cell (ps)", fontsize=Y_LABEL_FONT_SIZE)
        plt.ylim(bottom=0)
        plt.legend([LegendNames.get(alg) for alg in self.algs], fontsize=LEGEND_FONT_SIZE, loc=self.position, bbox_to_anchor=self.bbox)

        out_path = os.path.join(GRAPH_DIR, self.PLOT_NAME + '.' + MODE)

        plt.tight_layout(pad=1.0)
        plt.savefig(out_path, format=MODE)


class CompareAlgsOnGrids:

    def __init__(self, results: ra.Results):
        self.results = results
        self.tested_grids = []
        self.algs = []
        self.base_algs = []
        self.data_loaders = []
        self.x_labels = []
        self.PLOT_NAME = 'compare_algs_on_data'

    def set_base_algs(self, algs):
        self.base_algs = algs
        return self

    def set_algs(self, algs):
        self.algs = algs
        return self

    def set_grid(self, grid):
        self.tested_grid = grid
        return self
    
    def set_data_loaders_with_labels(self, loaders_labels):
        self.data_loaders = [d for d, _ in loaders_labels]
        self.x_labels = [l for _, l in loaders_labels]
        return self

    def set_plot_name(self, name):
        self.PLOT_NAME = name
        return self

    def gen_graphs_bars_as_baseline(self):
        plt.figure(figsize=(8, 6))

        x = np.arange(len(self.data_loaders)) * 1.2
        bar_width = 0.5 / len(self.algs)

        bar_colors = ["#3B6790", "#23486A", "#77B254", "#5B913B"]
        bar_patterns = ["//", "\\\\", "..", "o"]

        for i, alg in enumerate(self._interleave(self.base_algs, self.algs)):
            y_vals = []
            for loader in self.data_loaders:
                is_base = alg in self.base_algs
                
                if is_base:
                    grid = [*self.tested_grid]
                else:
                    grid = [*self.tested_grid, *loader]

                exp = self.results.get_experiments_with([*alg, *grid])
                if not exp:
                    y_vals.append(0)
                    continue

                measurements = exp[0].get_measurement_set()
                time = measurements.get_median(lambda m: m.compute_runtime_per_cell_per_iter())
                if time is None:
                    time = 0

                y_vals.append(time)

            y_vals = [from_ms_to_pico_seconds(v) for v in y_vals]
            plt.bar(
                x + i * bar_width - bar_width / 2,
                y_vals,
                bar_width,
                label=LegendNames.get(alg),
                color=bar_colors[i % len(bar_colors)],
                hatch=bar_patterns[i % len(bar_patterns)],
                edgecolor='white' 
            )
            

        plt.xticks(x + bar_width * (len(self.algs) / 2), self.x_labels, rotation=0, fontsize=X_TICKS_FONT_SIZE, y=X_TICKS_OFFSET)
        # plt.xlabel("Cases on grid " + self.tested_grid[0][1], fontsize=X_LABEL_FONT_SIZE)
        plt.ylabel("Time per one cell (ps)", fontsize=Y_LABEL_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        out_path = os.path.join(GRAPH_DIR, self.PLOT_NAME + '.' + MODE)
        plt.tight_layout(pad=1.0)
        plt.savefig(out_path, format=MODE)

    def gen_graphs_lines_as_base_lines(self):
        plt.figure(figsize=(8, 6))

        x = np.arange(len(self.data_loaders)) * 1.2
        bar_width = 0.5 / len(self.algs)

        # bar_colors = ["#3B6790", "#23486A", "#77B254", "#5B913B"]

        bar_colors = ["#3B6790", "#77B254"]
        base_line_colors = ["tab:blue", "tab:green"]

        # bar_patterns = ["//", "\\\\", "..", "o"]
        bar_patterns = ['//', ".."]
        ax_pattern = ['-', '--', '-.', ':']

        for i, alg in enumerate(self.algs):
            y_vals = []
            
            base_exp = results.get_experiments_with(
                [*self.base_algs[i], *self.tested_grid])[0]

            base_alg_value = base_exp.get_measurement_set().get_median(
                lambda m: m.compute_runtime_per_cell_per_iter())
            
            base_alg_value = from_ms_to_pico_seconds(base_alg_value)
            
            plt.axhline(
                y=base_alg_value, color=base_line_colors[i],
                linestyle=ax_pattern[i % len(ax_pattern)],
                label=LegendNames.get(self.base_algs[i]) + ' (base)')

            for loader in self.data_loaders:
                is_base = alg in self.base_algs
                
                if is_base:
                    grid = [*self.tested_grid]
                else:
                    grid = [*self.tested_grid, *loader]

                exp = self.results.get_experiments_with([*alg, *grid])
                if not exp:
                    y_vals.append(0)
                    continue

                measurements = exp[0].get_measurement_set()
                time = measurements.get_median(lambda m: m.compute_runtime_per_cell_per_iter())
                if time is None:
                    time = 0

                y_vals.append(time)

            y_vals = [from_ms_to_pico_seconds(v) for v in y_vals]
            plt.bar(
                x + i * bar_width - bar_width / 2,
                y_vals,
                bar_width,
                label=LegendNames.get(alg),
                color=bar_colors[i % len(bar_colors)],
                hatch=bar_patterns[i % len(bar_patterns)],
                edgecolor='white' 
            )
            

        plt.xticks(x, self.x_labels, rotation=0, fontsize=X_TICKS_FONT_SIZE, y=X_TICKS_OFFSET)
        # plt.xlabel("Cases on grid " + self.tested_grid[0][1], fontsize=X_LABEL_FONT_SIZE)
        plt.ylabel("Time per one cell (ps)", fontsize=Y_LABEL_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE, loc='upper right', bbox_to_anchor=(1.0, 0.85))
        out_path = os.path.join(GRAPH_DIR, self.PLOT_NAME + '.' + MODE)
        plt.tight_layout(pad=1.0)
        plt.savefig(out_path, format=MODE)

    def _interleave(self, a, b):
        return [val for pair in zip(a, b) for val in pair]            

def combined(alg, data):
    return [*alg, *data]

def print_line_graph(
    results: ra.Results,
    plot_name: str,
    algs_with_colors_etcs,
    position=None,
    bbox=None,
):
    algs = [alg for alg, _, _, _ in algs_with_colors_etcs]
    colors = [color for _, color, _, _ in algs_with_colors_etcs]
    markers = [marker for _, _, marker, _ in algs_with_colors_etcs]
    linestyles = [linestyle for _, _, _, linestyle in algs_with_colors_etcs]

    TimePerCellPerIter__InputSize(results) \
        .set_algs(algs) \
        .set_position(position) \
        .set_bbox(bbox) \
        .set_colors(colors) \
        .set_markers(markers) \
        .set_linestyles(linestyles) \
        .set_plot_name(plot_name) \
        .set_grids([
            ALG_LIST.g_2048,
            ALG_LIST.g_4096,
            ALG_LIST.g_8192,
            ALG_LIST.g_16384,
        ]) \
        .gen_graphs()

def print_bar_plot(results, plot_name):
    setup = CompareAlgsOnGrids(results) \
        .set_plot_name(plot_name) \
        .set_base_algs([
            # ALG_LIST.cuda_naive_bitwise_cols_32,
            ALG_LIST.cuda_naive_bitwise_cols_64,

            # ALG_LIST.cuda_naive_bitwise_tiles_32,
            ALG_LIST.cuda_naive_bitwise_tiles_64,
        ]) \
        .set_algs([

            # ALG_LIST.cuda_local_one_cell_cols_32,
            ALG_LIST.cuda_local_one_cell_cols_64,

            # ALG_LIST.cuda_local_one_cell_bit_tiles_32,
            ALG_LIST.cuda_local_one_cell_bit_tiles_64,
        ]) \
        .set_data_loaders_with_labels([
            (ALG_LIST.data__full_work, 'Busy 100%'),
            (ALG_LIST.data__66_work, 'Busy 66%'),
            (ALG_LIST.data__33_work, 'Busy 33%'),
            (ALG_LIST.data__no_work, 'Busy 0%'),
        ]) \
        .set_grid(
            ALG_LIST.g_16384,
        )
    
    setup.set_plot_name(plot_name + '__bars').gen_graphs_bars_as_baseline()
    setup.set_plot_name(plot_name + '__lines').gen_graphs_lines_as_base_lines()
    
def print_stats(results):
    grid = '16384x16384'

    naive_char = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'char'), (ra.Key.grid_dimensions, grid)])[0]
    naive_char_val = naive_char.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    naive_int = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-naive'), (ra.Key.base_grid_encoding, 'int'), (ra.Key.grid_dimensions, grid)])[0]
    naive_int_val = naive_int.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    sota_packed_32 = results.get_experiments_with([(ra.Key.algorithm_name, 'eff-sota-packed-32'), (ra.Key.grid_dimensions, grid)])[0]
    sota_packed_32_val = sota_packed_32.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    and5 = results.get_experiments_with([(ra.Key.algorithm_name, 'an5d'), (ra.Key.grid_dimensions, grid)])[0]
    and5_val = and5.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    bitwise_tiles_64 = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-tiles-64'), (ra.Key.grid_dimensions, grid)])[0]
    bitwise_tiles_64_val = bitwise_tiles_64.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    bitwise_cols_64 = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-naive-bitwise-cols-64'), (ra.Key.grid_dimensions, grid)])[0]
    bitwise_cols_64_val = bitwise_cols_64.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    local_tiles_66 = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-64--bit-tiles'), (ra.Key.tag, 'no-work'), (ra.Key.grid_dimensions, grid)])[0]
    local_tiles_66_val = local_tiles_66.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    local_tiles_full_work = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-64--bit-tiles'), (ra.Key.tag, 'full-work'), (ra.Key.grid_dimensions, grid)])[0]
    local_tiles_full_work_val = local_tiles_full_work.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    local_cols_full_work = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-64'), (ra.Key.tag, 'full-work'), (ra.Key.grid_dimensions, grid)])[0]
    local_cols_full_work_val = local_cols_full_work.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    local_tiles_no_work = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-64--bit-tiles'), (ra.Key.tag, 'no-work'), (ra.Key.grid_dimensions, grid)])[0]
    local_tiles_no_work_val = local_tiles_no_work.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())

    local_cols_no_work = results.get_experiments_with([(ra.Key.algorithm_name, 'gol-cuda-local-one-cell-cols-64'), (ra.Key.tag, 'no-work'), (ra.Key.grid_dimensions, grid)])[0]
    local_cols_no_work_val = local_cols_no_work.get_measurement_set().get_mean(lambda m: m.compute_runtime_per_cell_per_iter())
    
    print(f'base case (no work reduction) speedup over best naive: {naive_char_val / bitwise_tiles_64_val:.2f} x')
    print(f'base case (no work reduction) speedup over best sota: {sota_packed_32_val / bitwise_tiles_64_val:.2f} x')
    print(f'base case (no work reduction) speedup over AN5D: {and5_val / bitwise_tiles_64_val:.2f} x')

    print(f'work reduction (no work) speedup over best naive: {naive_char_val / local_tiles_66_val:.2f} x')
    print(f'work reduction (no work) speedup over best sota: {sota_packed_32_val / local_tiles_66_val:.2f} x')
    print(f'work reduction (no work) speedup over AN5D: {and5_val / local_tiles_66_val:.2f} x')

    print(f'work reduction tiles (full work) slowdown over base: {local_tiles_full_work_val / bitwise_tiles_64_val:.2f} x')
    print(f'work reduction cols (full work) slowdown over base: {local_cols_full_work_val / bitwise_cols_64_val:.2f} x')

    print(f'work reduction tiles (no work) speedup over base: {bitwise_tiles_64_val / local_tiles_no_work_val:.2f} x')
    print(f'work reduction cols (no work) speedup over base: {bitwise_cols_64_val / local_cols_no_work_val:.2f} x')

    print(f'best "core" over baseline: {naive_char_val / bitwise_tiles_64_val:.2f}')
    print(f'best "core" over sota: {sota_packed_32_val / bitwise_tiles_64_val:.2f}')
    
    print(f'best "optimized" over baseline: {naive_char_val / local_tiles_no_work_val:.2f}')
    print(f'best "optimized" over sota: {sota_packed_32_val / local_tiles_no_work_val:.2f}')

    print(f'best "core" over best "an5d": {and5_val / bitwise_tiles_64_val:.2f}')
    print(f'best "optimized" over best "an5d": {and5_val / local_tiles_no_work_val:.2f}')

    print(f'an5d over char baseline: {naive_char_val / and5_val:.2f}')
    print(f'an5d over int baseline: {naive_int_val / and5_val:.2f}')




for architecture in ARCHITECTURES:
    DATA_DIR = BASE_DIR.format(architecture=architecture)
    results = ra.Results.from_directory(DATA_DIR)

    print_line_graph(
        results,
        plot_name=f'{architecture}__ours_cols_tiles_vs_sota',
        algs_with_colors_etcs=[

            (ALG_LIST.eff_sota_packed_32, 'tab:red', '<', '--'), 

            (ALG_LIST.cuda_naive_bitwise_cols_32, 'tab:blue', 'd', ':'),  
            (ALG_LIST.cuda_naive_bitwise_cols_64, 'tab:blue', 's', '-'),  
            
            (ALG_LIST.cuda_naive_bitwise_tiles_32, 'tab:green', 'o', ':'),
            (ALG_LIST.cuda_naive_bitwise_tiles_64, 'tab:green', '^', '-'),
        ])

    print_line_graph(
        results,
        plot_name=f'{architecture}__final_comparison',
        position='center right',
        bbox=(1.0, 0.45),
        algs_with_colors_etcs=[

            (ALG_LIST.cuda_naive_int, 'tab:pink', 's', '--'),   
            (ALG_LIST.cuda_naive_char, 'tab:blue', 'o', '-'),   

            (ALG_LIST.cuda_an5d, 'tab:orange', 'd', '-.'), 
            
            (ALG_LIST.eff_sota_packed_32, 'tab:red', '<', '-.'), 
            
            (ALG_LIST.cuda_naive_bitwise_tiles_64, 'tab:green', '^', '-'), 

            (combined(ALG_LIST.cuda_local_one_cell_bit_tiles_64, ALG_LIST.data__no_work), 'tab:blue', '*', '-'), 
        ])

    print_bar_plot(results, f'{architecture}__algs_on_different_data')

    print ('Stats for:', architecture)
    print_stats(results)

    print()

# debug graph
# exit()

USE_DEBUG_NAMES = True

for architecture in ARCHITECTURES:
    DATA_DIR = BASE_DIR.format(architecture=architecture)
    results = ra.Results.from_directory(DATA_DIR)

    print_line_graph(
        results,
        plot_name=f'__tmp_working_line_graph_{architecture}',
        algs_with_colors_etcs=[
            (ALG_LIST.cuda_naive_char, 'blue', 'o', '-'),
            (ALG_LIST.cuda_naive_int,  'red', 's', '-.'),

            (ALG_LIST.cuda_an5d, 'green', 'v', '--'),
            
            (ALG_LIST.cuda_naive_bitwise_cols_32, 'purple', 'x', '--'),
            (ALG_LIST.cuda_naive_bitwise_cols_64, 'orange', 'd', ':'),
            (ALG_LIST.cuda_naive_bitwise_tiles_32, 'black', 'p', '-'),
            (ALG_LIST.cuda_naive_bitwise_tiles_64, 'brown', 'h', '-.'),
            
            (ALG_LIST.eff_baseline_shm, 'red', 's', '-.'),
            (ALG_LIST.eff_sota_packed_32, 'green', 'v', 'dotted'), 
            (ALG_LIST.eff_sota_packed_64, 'blue', 'o', 'dotted'), 
        ])
