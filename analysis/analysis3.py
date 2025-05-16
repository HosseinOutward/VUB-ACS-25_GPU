import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)

def run_cpp_analysis(executable_path, batch_count_arg, execution_cwd):
    command = [executable_path, str(batch_count_arg)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=execution_cwd)
    return result.stdout


def parse_output(output_str: str) -> dict:
    timings = {}
    for line in output_str.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # We primarily need OverallBatchProcessingTime_ms and ActualBatchCount
            # FilterWrite times can be kept if present, but not essential for the comparison plot
            if "OverallBatchProcessingTime_ms" in key or \
               "ActualBatchCount" in key or \
               "FilterWriteQueueTime_ns" in key or \
               "FilterWriteExecTime_ns" in key or \
               "DeviceProfilingTimerResolution_ns" in key:
                timings[key] = value
            # Individual batch overall times are not needed for this specific comparison plot
            # elif "Batch_" in key and "OverallTime_ms" in key:
            #     timings[key] = value


    raw_actual_batches = timings.get("ActualBatchCount", "0")
    try:
        actual_batches = int(raw_actual_batches)
    except ValueError:
        print(f"Warning: Could not parse ActualBatchCount '{raw_actual_batches}' as int. Defaulting to 0.")
        actual_batches = 0
    timings["ActualBatchCount"] = actual_batches

    # REMOVED: Initialization and population of detailed batch_..._overall_ms lists
    # REMOVED: Incremental calculation for overall_batch_keys

    return timings


def ns_to_ms(ns_val):
    return ns_val / 1_000_000.0


def plot_timings(concurrent_run_data, sequential_run_data, batch_sizes_used):
    """
    Plots the overall C++ processing time for concurrent and sequential designs
    against the number of batches for multiple runs.

    Args:
        concurrent_run_data (list of dict): Parsed timings for the concurrent design.
        sequential_run_data (list of dict): Parsed timings for the sequential design.
        batch_sizes_used (list of int): The actual batch sizes used for the runs.
    """
    if not concurrent_run_data and not sequential_run_data:
        print("No data to plot.")
        return

    concurrent_batch_counts = [data.get("ActualBatchCount", 0) for data in concurrent_run_data]
    concurrent_overall_times_ms = [float(data.get("OverallBatchProcessingTime_ms", 0)) for data in concurrent_run_data]
    
    sequential_batch_counts = [data.get("ActualBatchCount", 0) for data in sequential_run_data]
    sequential_overall_times_ms = [float(data.get("OverallBatchProcessingTime_ms", 0)) for data in sequential_run_data]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    if concurrent_run_data:
        ax.plot(concurrent_batch_counts, concurrent_overall_times_ms, marker='o', linestyle='-', label='Concurrent Design (analysis2)')
    if sequential_run_data:
        ax.plot(sequential_batch_counts, sequential_overall_times_ms, marker='x', linestyle='--', label='Sequential Design (analysis3)')

    ax.set_xlabel('Number of Batches')
    ax.set_ylabel('Overall Processing Time (ms)')
    ax.set_title('Performance Comparison: Concurrent vs. Sequential Design', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Ensure x-axis ticks match the batch sizes used if they are consistent
    # This helps if some batch sizes were skipped or failed for one executable
    if batch_sizes_used:
        ax.set_xticks(batch_sizes_used)
        ax.set_xticklabels([str(bs) for bs in batch_sizes_used], rotation=45, ha="right")


    plt.tight_layout()
    plt.show()


def main():
    concurrent_exe_name = "conv_2d_faster_analysis2.exe"
    sequential_exe_name = "conv_2d_faster_analysis3.exe"
    
    # Define batch sizes: start small, then increase more rapidly towards max
    batch_sizes_to_test = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450] # Max 250 as per original request
    # max_batch_size = 250 
    
    # batch_sizes_to_test = sorted(list(set(bs for bs in batch_sizes_to_test if bs <= max_batch_size)))
    # if not batch_sizes_to_test or batch_sizes_to_test[-1] < max_batch_size:
    #     if max_batch_size not in batch_sizes_to_test:
    #          batch_sizes_to_test.append(max_batch_size)
    # batch_sizes_to_test = sorted(list(set(batch_sizes_to_test)))


    concurrent_run_data = []
    sequential_run_data = []
    
    processed_batch_sizes = []


    print(f"--- Comparing: {concurrent_exe_name} (Concurrent) vs. {sequential_exe_name} (Sequential) ---")

    for batch_count_arg in batch_sizes_to_test:
        print(f"\n--- Testing Batch Count: {batch_count_arg} ---")
        current_batch_processed = False

        # Run Concurrent Design
        executable_path_concurrent = os.path.join(WORKSPACE_ROOT, "bin", concurrent_exe_name)
        if os.path.exists(executable_path_concurrent):
            print(f"  Running {concurrent_exe_name}...")
            cpp_output_concurrent = run_cpp_analysis(executable_path_concurrent, batch_count_arg, WORKSPACE_ROOT)
            parsed_timings_concurrent = parse_output(cpp_output_concurrent)
            if parsed_timings_concurrent.get("ActualBatchCount") == batch_count_arg: # Basic check
                concurrent_run_data.append(parsed_timings_concurrent)
                print(f"    {concurrent_exe_name} Overall Time: {parsed_timings_concurrent.get('OverallBatchProcessingTime_ms', 'N/A')} ms")
                current_batch_processed = True
            else:
                print(f"    Warning: {concurrent_exe_name} did not return expected batch count for {batch_count_arg}. Skipping this data point.")
        else:
            print(f"    Error: Executable not found - {executable_path_concurrent}")

        # Run Sequential Design
        executable_path_sequential = os.path.join(WORKSPACE_ROOT, "bin", sequential_exe_name)
        if os.path.exists(executable_path_sequential):
            print(f"  Running {sequential_exe_name}...")
            cpp_output_sequential = run_cpp_analysis(executable_path_sequential, batch_count_arg, WORKSPACE_ROOT)
            parsed_timings_sequential = parse_output(cpp_output_sequential)
            if parsed_timings_sequential.get("ActualBatchCount") == batch_count_arg: # Basic check
                sequential_run_data.append(parsed_timings_sequential)
                print(f"    {sequential_exe_name} Overall Time: {parsed_timings_sequential.get('OverallBatchProcessingTime_ms', 'N/A')} ms")
                current_batch_processed = True # Even if only one processed, consider the batch size tested
            else:
                print(f"    Warning: {sequential_exe_name} did not return expected batch count for {batch_count_arg}. Skipping this data point.")
        else:
            print(f"    Error: Executable not found - {executable_path_sequential}")
        
        if current_batch_processed:
            processed_batch_sizes.append(batch_count_arg)


    plot_timings(concurrent_run_data, sequential_run_data, processed_batch_sizes)


if __name__ == "__main__":
    main()
