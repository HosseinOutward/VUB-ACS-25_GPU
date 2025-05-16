import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import json # Added for JSON output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

def run_cpp_analysis(executable_path, batch_count_arg, kernel_size_arg, execution_cwd):
    command = [executable_path, str(batch_count_arg), str(kernel_size_arg)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=execution_cwd)
    return result.stdout


def parse_output(output_str: str) -> dict:
    timings = {}
    for line in output_str.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if "OverallBatchProcessingTime_ms" in key or \
               "ActualBatchCount" in key or \
               "KernelSize" in key or \
               "Error" in key: # Capture error messages
                timings[key] = value

    raw_actual_batches = timings.get("ActualBatchCount", "0")
    try:
        actual_batches = int(raw_actual_batches)
    except ValueError:
        print(f"Warning: Could not parse ActualBatchCount '{raw_actual_batches}' as int. Defaulting to 0.")
        actual_batches = 0
    timings["ActualBatchCount"] = actual_batches

    raw_kernel_size = timings.get("KernelSize", "0")
    try:
        kernel_size = int(raw_kernel_size)
    except ValueError:
        print(f"Warning: Could not parse KernelSize '{raw_kernel_size}' as int. Defaulting to 0.")
        kernel_size = 0
    timings["KernelSize"] = kernel_size

    return timings


def ns_to_ms(ns_val):
    return ns_val / 1_000_000.0


def plot_timings(kernel_size_data, target_batch_count):
    """
    Plots the average processing time per batch against kernel size.

    Args:
        kernel_size_data (dict): Keys are kernel sizes, values are dicts with 'times_ms' and 'batch_counts'.
        target_batch_count (int): The batch count used for averaging.
    """
    if not kernel_size_data:
        print("No data to plot.")
        return

    kernel_sizes = sorted(kernel_size_data.keys())
    avg_times_per_batch = []

    for ks in kernel_sizes:
        data = kernel_size_data[ks]
        total_time_ms = sum(data.get('times_ms', []))
        total_batches = sum(data.get('batch_counts', []))
        
        if total_batches > 0:
            # Calculate average time for the entire run, then divide by target_batch_count for per-batch average
            # This assumes the C++ output 'OverallBatchProcessingTime_ms' is for the *entire* batch_count run.
            avg_overall_time_ms = total_time_ms / len(data.get('times_ms', [1])) # Avg if multiple runs for same kernel size (not current case)
            if target_batch_count > 0:
                 avg_time_per_single_batch = avg_overall_time_ms / target_batch_count
            else:
                avg_time_per_single_batch = 0
            avg_times_per_batch.append(avg_time_per_single_batch)
            print(f"Kernel Size {ks}: Total Time {total_time_ms:.2f}ms, Total Batches {total_batches}, Avg Time/Batch: {avg_time_per_single_batch:.4f}ms")
        else:
            avg_times_per_batch.append(0)
            print(f"Kernel Size {ks}: No successful runs or zero batches processed.")

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    ax.plot(kernel_sizes, avg_times_per_batch, marker='o', linestyle='-')

    ax.set_xlabel('Kernel Size (N x N)')
    ax.set_ylabel(f'Average Processing Time per Batch (ms) (for {target_batch_count} batches)')
    ax.set_title('Effect of Kernel Size on Processing Time per Batch', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    ax.set_xticks(kernel_sizes)
    ax.set_xticklabels([str(ks) for ks in kernel_sizes], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def save_kernel_size_data_to_file(kernel_size_data, fixed_batch_count, filename_prefix="analysis4_kernel_size"):
    """Saves the kernel size timing data to a JSON file."""
    filename = f"{filename_prefix}_batch_{fixed_batch_count}_timings.json"
    filepath = os.path.join(SCRIPT_DIR, filename)
    data_to_save = {
        "fixed_batch_count": fixed_batch_count,
        "kernel_size_runs": kernel_size_data
    }
    try:
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Kernel size data saved to {filepath}")
    except IOError as e:
        print(f"Error saving kernel size data to {filepath}: {e}")

def main():
    # Use the analysis4 executable, assuming it's the one modified for kernel size input
    cpp_executable_name = "conv_2d_faster_analysis4.exe" 
    executable_path = os.path.join(WORKSPACE_ROOT, "bin", cpp_executable_name)

    if not os.path.exists(executable_path):
        print(f"Error: Executable not found - {executable_path}")
        print("Please build the C++ project first (e.g., conv_2d_faster_analysis4.cpp).")
        return

    kernel_sizes_to_test = [3, 6, 12, 24, 36, 50, 100, 150]
    fixed_batch_count = 150

    # Store data as: {kernel_size: {'times_ms': [...], 'batch_counts': [...]}
    all_run_data = {ks: {'times_ms': [], 'batch_counts': []} for ks in kernel_sizes_to_test}

    print(f"--- Analyzing {cpp_executable_name} with Batch Count: {fixed_batch_count} ---")

    for kernel_size in kernel_sizes_to_test:
        print(f"\n--- Testing Kernel Size: {kernel_size}x{kernel_size} ---")

        if not os.path.exists(executable_path):
            print(f"    Error: Executable not found - {executable_path}. Skipping kernel size {kernel_size}.")
            continue

        print(f"  Running {cpp_executable_name} with kernel {kernel_size}x{kernel_size}...")
        cpp_output = run_cpp_analysis(executable_path, fixed_batch_count, kernel_size, WORKSPACE_ROOT)
        parsed_timings = parse_output(cpp_output)

        if "Error" in parsed_timings:
            print(f"    Error from C++ for kernel {kernel_size}: {parsed_timings['Error']}")
            print(f"    Skipping data point for kernel size {kernel_size}.")
            continue

        actual_batches = parsed_timings.get("ActualBatchCount")
        returned_kernel_size = parsed_timings.get("KernelSize")
        overall_time_ms_str = parsed_timings.get("OverallBatchProcessingTime_ms")

        if actual_batches == fixed_batch_count and returned_kernel_size == kernel_size and overall_time_ms_str is not None:
            try:
                overall_time_ms = float(overall_time_ms_str)
                all_run_data[kernel_size]['times_ms'].append(overall_time_ms)
                all_run_data[kernel_size]['batch_counts'].append(actual_batches)
                avg_time_per_batch = overall_time_ms / fixed_batch_count if fixed_batch_count > 0 else 0
                print(f"    {cpp_executable_name} (Kernel {kernel_size}x{kernel_size}): Overall Time: {overall_time_ms:.2f} ms, Avg Time/Batch: {avg_time_per_batch:.4f} ms")
            except ValueError:
                print(f"    Warning: Could not parse OverallBatchProcessingTime_ms '{overall_time_ms_str}' as float for kernel {kernel_size}. Skipping.")
        else:
            print(f"    Warning: Data mismatch or missing for kernel {kernel_size}.")
            print(f"      Expected Batches: {fixed_batch_count}, Got: {actual_batches}")
            print(f"      Expected Kernel: {kernel_size}, Got: {returned_kernel_size}")
            print(f"      Overall Time: {overall_time_ms_str}")
            print(f"    Skipping this data point for kernel size {kernel_size}.")

    plot_timings(all_run_data, fixed_batch_count)
    save_kernel_size_data_to_file(all_run_data, fixed_batch_count, cpp_executable_name.replace('.exe', ''))


if __name__ == "__main__":
    main()
