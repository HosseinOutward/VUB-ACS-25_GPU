import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
from analysis import generate_data
from utilities import theoretical_info, initialize_device_info

# Initialize device info and global parameters
device_info = initialize_device_info('../bin/device_info')

stride = 2
single_batch_size = 8
batch_counts = 15

# Configuration for GPU kernel benchmarking
class BenchConfig:
    def __init__(self):
        self.kernel = "sum"
        self.float_type = "float"

config = BenchConfig()
exe_path = "../bin/gpu_benchmark"  # Adjust path as needed

def run_bench(kernel, sizes, repetitions, float_type, runs_per_size, exe,
              make_plots=False, verbose=False, elements_per_thread=1):
    """
    Run GPU benchmark for specified kernel and parameters.

    Args:
        kernel: Kernel name (e.g., 'sum', 'fma', 'sum_overlap')
        sizes: List of array sizes to test
        repetitions: Number of repetitions per size
        float_type: Data type ('float' or 'double')
        runs_per_size: Number of runs per size
        exe: Path to benchmark executable
        make_plots: Whether to generate plots
        verbose: Verbose output
        elements_per_thread: Elements processed per thread

    Returns:
        dict: Benchmark results with 'aggregates' and 'results' keys
    """
    try:
        results = {'aggregates': [], 'results': []}

        for size in sizes:
            cmd = [exe, kernel, str(size), str(repetitions)]

            # Add type argument - handle both string and int formats
            if isinstance(float_type, int):
                if float_type == 32:
                    cmd.append("--type=float")
                elif float_type == 64:
                    cmd.append("--type=double")
            else:
                cmd.append(f"--type={float_type}")

            cmd.append(f"--elements-per-thread={elements_per_thread}")

            if verbose:
                print(f"Running: {' '.join(cmd)}")

            # Run benchmark
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                # Parse output - look for performance metrics
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'GOPS' in line or 'Performance' in line:
                        # Try to extract numeric value
                        try:
                            # Look for patterns like "123.45 GOPS" or "Performance: 123.45"
                            import re
                            numbers = re.findall(r'(\d+\.?\d*)', line)
                            if numbers:
                                gops_mean = float(numbers[0])
                                results['aggregates'].append({
                                    'size': size,
                                    'gops_mean': gops_mean,
                                    'kernel': kernel
                                })
                                results['results'].append({
                                    'size': size,
                                    'performance': gops_mean,
                                    'kernel': kernel
                                })
                                break
                        except:
                            continue

                # If no GOPS found, try to extract any timing information
                if not results['aggregates']:
                    for line in lines:
                        if 'time' in line.lower() or 'ms' in line:
                            try:
                                import re
                                numbers = re.findall(r'(\d+\.?\d*)', line)
                                if numbers:
                                    time_ms = float(numbers[0])
                                    # Convert to rough GOPS estimate (size * operations / time)
                                    ops_per_element = 1  # Assume 1 operation per element
                                    gops = (size * ops_per_element) / (time_ms * 1e6)
                                    results['aggregates'].append({
                                        'size': size,
                                        'gops_mean': gops,
                                        'kernel': kernel
                                    })
                                    results['results'].append({
                                        'size': size,
                                        'performance': gops,
                                        'kernel': kernel
                                    })
                                    break
                            except:
                                continue
            else:
                if verbose:
                    print(f"Benchmark failed: {result.stderr}")

        return results

    except Exception as e:
        if verbose:
            print(f"Error running benchmark: {e}")
        return {'aggregates': [], 'results': []}

def breakdown_time(data):
    """
    Break down timing data into compute, memory, and etc time components.

    Args:
        data: Dictionary with timing data including 'Upload_time', 'Kernel_time', 'Download_time'
              Each contains multiple samples, and each sample contains multiple batches.

    Returns:
        dict: Contains 'compute_time', 'memory_time', 'etc_time', 'total_time'
    """
    # Find the sample with median overall time
    overall_times = data['gpu_overall_time_0']
    median_overall_time = np.median(overall_times)

    # Find which sample has the overall time closest to the median
    median_sample_idx = np.argmin(np.abs(np.array(overall_times) - median_overall_time))

    # Get the timing data for that specific sample and sum up all batches
    upload_time = np.sum(data['Upload_time'][median_sample_idx])
    kernel_time = np.sum(data['Kernel_time'][median_sample_idx])
    download_time = np.sum(data['Download_time'][median_sample_idx])
    total_time = overall_times[median_sample_idx]

    compute_time = kernel_time
    memory_time = upload_time + download_time
    etc_time = total_time - compute_time - memory_time

    return {
        'compute_time': compute_time,
        'memory_time': memory_time*0,
        'etc_time': etc_time*0,
        'total_time': total_time*0+compute_time
    }

def create_stacked_plot(x_values, measured_data, theoretical_data, x_label, title):
    """
    Create a stacked plot showing time breakdown for measured and theoretical data.

    Args:
        x_values: X-axis values
        measured_data: List of breakdown dictionaries from breakdown_time()
        theoretical_data: List of theoretical timing dictionaries
        x_label: Label for x-axis
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract measured time components
    measured_etc = [d['etc_time'] for d in measured_data]
    measured_memory = [d['memory_time'] for d in measured_data]
    measured_compute = [d['compute_time'] for d in measured_data]
    measured_total = [d['total_time'] for d in measured_data]

    # Extract theoretical time components
    theo_memory = [d['Time-spent memory'] for d in theoretical_data]
    theo_compute = [d['Time-spent compute'] for d in theoretical_data]
    theo_total = [d['Time-spent overall'] for d in theoretical_data]

    # Plot measured data as stacked bars
    width = 0.35
    x_pos = np.arange(len(x_values))

    # Measured stacked components
    ax.bar(x_pos - width/2, measured_etc, width, label='Measured - Etc', color='lightcoral', alpha=0.8)
    ax.bar(x_pos - width/2, measured_memory, width, bottom=measured_etc,
           label='Measured - Memory', color='lightblue', alpha=0.8)
    ax.bar(x_pos - width/2, measured_compute, width,
           bottom=np.array(measured_etc) + np.array(measured_memory),
           label='Measured - Compute', color='lightgreen', alpha=0.8)

    # Theoretical stacked components (different shades)
    ax.bar(x_pos + width/2, theo_memory, width, label='Theoretical - Memory',
           color='darkblue', alpha=0.6)
    ax.bar(x_pos + width/2, theo_compute, width, bottom=theo_memory,
           label='Theoretical - Compute', color='darkgreen', alpha=0.6)

    # Overall time lines
    ax.plot(x_pos - width/2, measured_total, 'ro-', label='Measured Total', linewidth=2)
    ax.plot(x_pos + width/2, theo_total, 'bo-', label='Theoretical Total', linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Time (ms)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def analyze_kernel_size_performance():
    """Analyze performance vs kernel size."""
    print("Analyzing performance vs kernel size...")

    input_x_y_c = (500, 500, 20)  # Fixed input size
    kernel_sizes = [(3, 3, 5), (5, 5, 5), (7, 7, 5), (9, 9, 5), (11, 11, 5)]

    measured_breakdowns = []
    theoretical_data = []
    kernel_labels = []

    for kernel_x_y_fc in kernel_sizes:
        print(f"  Processing kernel size: {kernel_x_y_fc}")

        # Get measured data
        data = generate_data(
            kernel_arg="%d,%d,%d" % kernel_x_y_fc,
            input_arg="%d,%d,%d" % input_x_y_c,
            samples=3, single_batch_size=single_batch_size,
            batch_counts=batch_counts, cpu_or_gpu='gpu', cpu_threads=0,
            stride=stride, gpu_detailed_timing=True,
            gpu_local_mem=False, gpu_pinned_mem=False, gpu_full_duplex=False
        )

        breakdown = breakdown_time(data)
        measured_breakdowns.append(breakdown)

        # Get theoretical data
        theo_data = theoretical_info(input_x_y_c, kernel_x_y_fc, device_info,
                                   stride, single_batch_size, batch_counts)
        theoretical_data.append(theo_data)

        kernel_labels.append(f"{kernel_x_y_fc[0]}x{kernel_x_y_fc[1]}")

    # Create plot
    fig = create_stacked_plot(kernel_labels, measured_breakdowns, theoretical_data,
                             "Kernel Size", "Performance vs Kernel Size")
    plt.savefig('performance_vs_kernel_size_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

    return measured_breakdowns, theoretical_data

def analyze_input_size_performance():
    """Analyze performance vs input size."""
    print("Analyzing performance vs input size...")

    kernel_x_y_fc = (7, 7, 5)  # Fixed kernel size
    input_sizes = [200, 400, 600, 800, 1000]

    measured_breakdowns = []
    theoretical_data = []

    for size in input_sizes:
        input_x_y_c = (size, size, 20)
        print(f"  Processing input size: {input_x_y_c}")

        # Get measured data
        data = generate_data(
            kernel_arg="%d,%d,%d" % kernel_x_y_fc,
            input_arg="%d,%d,%d" % input_x_y_c,
            samples=3, single_batch_size=single_batch_size,
            batch_counts=batch_counts, cpu_or_gpu='gpu', cpu_threads=0,
            stride=stride, gpu_detailed_timing=True,
            gpu_local_mem=False, gpu_pinned_mem=False, gpu_full_duplex=False
        )

        breakdown = breakdown_time(data)
        measured_breakdowns.append(breakdown)

        # Get theoretical data
        theo_data = theoretical_info(input_x_y_c, kernel_x_y_fc, device_info,
                                   stride, single_batch_size, batch_counts)
        theoretical_data.append(theo_data)

    # Create plot
    fig = create_stacked_plot(input_sizes, measured_breakdowns, theoretical_data,
                             "Input Size", "Performance vs Input Size")
    plt.savefig('performance_vs_input_size_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

    return measured_breakdowns, theoretical_data

def run_efficiency_comparison():
    """Compare efficient kernels with their inefficient counterparts"""
    comparison_sizes = [25000, 50000, 100000, 250000]

    print("🔄 Running efficient vs inefficient kernel comparison...")

    # Test pairs: (efficient, inefficient, description)
    kernel_pairs = [
        ("sum", "sum_overlap", "Overlapping Work"),
        ("sum", "sum_strided", "Strided Memory Access"),
        ("sum", "sum_reverse", "Branch Divergence"),
        ("sum", "sum_redundant", "Redundant Memory Reads"),
        ("fma", "fma_overlap", "FMA Overlapping Work"),
        ("fma", "fma_strided", "FMA Strided Access")
    ]

    results = {}

    for efficient, inefficient, description in kernel_pairs:
        print(f"\n📊 Testing: {description}")

        try:
            # Test efficient kernel
            print(f"   Running efficient {efficient}...")
            result_eff = run_bench(
                kernel=efficient,
                sizes=comparison_sizes,
                repetitions=50,
                float_type=config.float_type,
                runs_per_size=2,
                exe=exe_path,
                make_plots=False,
                verbose=False,
                elements_per_thread=4
            )

            # Test inefficient kernel
            print(f"   Running inefficient {inefficient}...")
            result_ineff = run_bench(
                kernel=inefficient,
                sizes=comparison_sizes,
                repetitions=50,
                float_type=config.float_type,
                runs_per_size=2,
                exe=exe_path,
                make_plots=False,
                verbose=False,
                elements_per_thread=4
            )

            if result_eff['aggregates'] and result_ineff['aggregates']:
                results[description] = {
                    'efficient': result_eff['aggregates'],
                    'inefficient': result_ineff['aggregates']
                }

                # Calculate average performance difference
                avg_eff = np.mean([r['gops_mean'] for r in result_eff['aggregates']])
                avg_ineff = np.mean([r['gops_mean'] for r in result_ineff['aggregates']])
                slowdown = avg_eff / avg_ineff if avg_ineff > 0 else float('inf')

                print(f"   Efficient: {avg_eff:.2f} GOPS")
                print(f"   Inefficient: {avg_ineff:.2f} GOPS")
                print(f"   Slowdown: {slowdown:.2f}x")

        except Exception as e:
            print(f"   Failed to compare {description}: {e}")

    if results:
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        plot_idx = 0
        for description, data in list(results.items())[:4]:  # Plot first 4 comparisons
            ax = axes[plot_idx]

            sizes = [r['size'] for r in data['efficient']]
            gops_eff = [r['gops_mean'] for r in data['efficient']]
            gops_ineff = [r['gops_mean'] for r in data['inefficient']]

            ax.plot(sizes, gops_eff, '-o', label='Efficient', linewidth=2, color='green')
            ax.plot(sizes, gops_ineff, '-s', label='Inefficient', linewidth=2, color='red')

            ax.set_xlabel('Array Size (elements)')
            ax.set_ylabel('Throughput (GOPS)')
            ax.set_title(f'{description}\nPerformance Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        plt.tight_layout()
        plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Slowdown summary plot
        plt.figure(figsize=(12, 6))

        categories = []
        slowdowns = []

        for description, data in results.items():
            sizes = [r['size'] for r in data['efficient']]

            for i, size in enumerate(sizes):
                eff_gops = data['efficient'][i]['gops_mean']
                ineff_gops = data['inefficient'][i]['gops_mean']
                slowdown = eff_gops / ineff_gops if ineff_gops > 0 else 0

                categories.append(f"{description}\n{size//1000}K")
                slowdowns.append(slowdown)

        # Create bar plot
        bars = plt.bar(range(len(categories)), slowdowns,
                      color=['red' if s > 2 else 'orange' if s > 1.5 else 'yellow' for s in slowdowns])

        plt.xlabel('Inefficiency Type and Array Size')
        plt.ylabel('Performance Degradation (Efficient/Inefficient)')
        plt.title('Performance Impact of Different Inefficiencies')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='No degradation')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, slowdown) in enumerate(zip(bars, slowdowns)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{slowdown:.1f}x', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig('inefficiency_impact.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print summary
        print(f"\n🏆 Inefficiency Impact Summary:")
        worst_slowdowns = {}
        for description, data in results.items():
            avg_eff = np.mean([r['gops_mean'] for r in data['efficient']])
            avg_ineff = np.mean([r['gops_mean'] for r in data['inefficient']])
            slowdown = avg_eff / avg_ineff if avg_ineff > 0 else float('inf')
            worst_slowdowns[description] = slowdown

        # Sort by worst performance impact
        sorted_inefficiencies = sorted(worst_slowdowns.items(), key=lambda x: x[1], reverse=True)

        for i, (description, slowdown) in enumerate(sorted_inefficiencies, 1):
            print(f"   {i}. {description}: {slowdown:.2f}x slower")

    return results

def run_elements_per_thread_comparison():
    """Compare performance with different elements per thread"""
    comparison_sizes = [50000, 100000, 250000, 500000]
    elements_per_thread_values = [1, 2, 4, 8, 16, 32]

    print("🔄 Running elements per thread comparison...")

    results = {}
    for ept in elements_per_thread_values:
        print(f"\n🧵 Testing {ept} elements per thread...")
        try:
            result = run_bench(
                kernel=config.kernel,
                sizes=comparison_sizes,
                repetitions=50,
                float_type=config.float_type,
                runs_per_size=2,
                exe=exe_path,
                make_plots=False,
                verbose=False,
                elements_per_thread=ept
            )

            if result['aggregates']:
                results[ept] = result['aggregates']
                avg_gops = np.mean([r['gops_mean'] for r in result['aggregates']])
                print(f"   Average performance: {avg_gops:.2f} GOPS")

        except Exception as e:
            print(f"   Failed with {ept} elements per thread: {e}")

    if len(results) >= 2:
        # Plot comparison
        plt.figure(figsize=(15, 5))

        # Performance vs elements per thread
        plt.subplot(1, 3, 1)
        for size in comparison_sizes:
            gops_values = []
            ept_values = []
            for ept, aggs in results.items():
                size_result = next((r for r in aggs if r['size'] == size), None)
                if size_result:
                    gops_values.append(size_result['gops_mean'])
                    ept_values.append(ept)

            if gops_values:
                plt.plot(ept_values, gops_values, '-o', label=f'{size:,} elements', linewidth=2)

        plt.xlabel('Elements per Thread')
        plt.ylabel('Throughput (GOPS)')
        plt.title('Performance vs Elements per Thread')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        # Thread efficiency
        plt.subplot(1, 3, 2)
        for size in comparison_sizes:
            efficiency_values = []
            ept_values = []
            for ept, aggs in results.items():
                size_result = next((r for r in aggs if r['size'] == size), None)
                if size_result:
                    num_threads = (size + ept - 1) // ept
                    efficiency = size / (num_threads * ept) * 100
                    efficiency_values.append(efficiency)
                    ept_values.append(ept)

            if efficiency_values:
                plt.plot(ept_values, efficiency_values, '-s', label=f'{size:,} elements', linewidth=2)

        plt.xlabel('Elements per Thread')
        plt.ylabel('Thread Efficiency (%)')
        plt.title('Thread Utilization Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        # Speedup relative to single element per thread
        plt.subplot(1, 3, 3)
        baseline_ept = min(results.keys())

        for size in comparison_sizes:
            speedup_values = []
            ept_values = []
            baseline_gops = None

            # Get baseline performance
            baseline_result = next((r for r in results[baseline_ept] if r['size'] == size), None)
            if baseline_result:
                baseline_gops = baseline_result['gops_mean']

            if baseline_gops:
                for ept, aggs in results.items():
                    size_result = next((r for r in aggs if r['size'] == size), None)
                    if size_result:
                        speedup = size_result['gops_mean'] / baseline_gops
                        speedup_values.append(speedup)
                        ept_values.append(ept)

                if speedup_values:
                    plt.plot(ept_values, speedup_values, '-^', label=f'{size:,} elements', linewidth=2)

        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Elements per Thread')
        plt.ylabel(f'Speedup vs {baseline_ept} elem/thread')
        plt.title('Speedup vs Single Element per Thread')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)

        plt.tight_layout()
        plt.savefig('elements_per_thread_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Find optimal elements per thread
        best_configs = {}
        for size in comparison_sizes:
            best_gops = 0
            best_ept = 1
            for ept, aggs in results.items():
                size_result = next((r for r in aggs if r['size'] == size), None)
                if size_result and size_result['gops_mean'] > best_gops:
                    best_gops = size_result['gops_mean']
                    best_ept = ept
            best_configs[size] = (best_ept, best_gops)

        print(f"\n🏆 Optimal elements per thread:")
        for size, (ept, gops) in best_configs.items():
            print(f"   {size:,} elements: {ept} elem/thread ({gops:.2f} GOPS)")

    return results

if __name__ == "__main__":
    # Run kernel size analysis
    kernel_measured, kernel_theoretical = analyze_kernel_size_performance()

    # Run input size analysis
    input_measured, input_theoretical = analyze_input_size_performance()

    # Run GPU kernel efficiency comparisons if executable exists
    if os.path.exists(exe_path):
        print("\n" + "="*60)
        print("🚀 Running GPU Kernel Performance Analysis")
        print("="*60)

        # Run efficiency comparison
        efficiency_results = run_efficiency_comparison()

        # Run elements per thread comparison
        elements_results = run_elements_per_thread_comparison()

        print("\n🎉 GPU kernel analysis complete!")
    else:
        print(f"\n⚠️  GPU benchmark executable not found at {exe_path}")
        print("Skipping GPU kernel efficiency analysis.")

    print("Analysis complete. Plots saved as PNG files.")
