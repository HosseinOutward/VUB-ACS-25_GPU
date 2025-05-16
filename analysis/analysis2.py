import subprocess
import re
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd # Added for DataFrame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
# EXECUTABLE_PATH = os.path.join(WORKSPACE_ROOT, "bin", "conv_2d_faster_analysis1.exe") # Will be set in main


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
            # Capture all _ns keys, including QueueTime_ns and ExecTime_ns
            # Keep FilterWrite times as they are singular, not per-batch
            if "FilterWriteQueueTime_ns" in key or \
               "FilterWriteExecTime_ns" in key or \
               "Resolution_ns" in key:
                timings[key] = value # Store as string initially
            elif "_ms" in key: # This will capture OverallBatchProcessingTime_ms and Batch_i_...OverallTime_ms
                timings[key] = value # Store as string initially
            elif "ActualBatchCount" in key:
                timings[key] = value # Store as string initially

    raw_actual_batches = timings.get("ActualBatchCount", "0")
    try:
        actual_batches = int(raw_actual_batches)
    except ValueError:
        print(f"Warning: Could not parse ActualBatchCount '{raw_actual_batches}' as int. Defaulting to 0.")
        actual_batches = 0
    timings["ActualBatchCount"] = actual_batches # Store back as int

    # Initialize lists for overall times for each batch operation type
    timings["batch_input_write_overall_ms"] = [0.0] * actual_batches
    timings["batch_kernel_overall_ms"] = [0.0] * actual_batches
    timings["batch_read_output_overall_ms"] = [0.0] * actual_batches

    def get_numeric_ns_value(raw_value):
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        if isinstance(raw_value, str) and raw_value.isdigit():
            return int(raw_value)
        return 0  # Default for None, non-digit strings, etc.

    def get_numeric_ms_value(raw_value):
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except ValueError:
                return 0.0 # Default for non-float strings
        return 0.0 # Default for None etc.

    for i in range(actual_batches):
        timings["batch_input_write_overall_ms"][i] = get_numeric_ms_value(timings.get(f"Batch_{i}_InputWriteOverallTime_ms"))
        timings["batch_kernel_overall_ms"][i] = get_numeric_ms_value(timings.get(f"Batch_{i}_KernelOverallTime_ms"))
        timings["batch_read_output_overall_ms"][i] = get_numeric_ms_value(timings.get(f"Batch_{i}_ReadOverallTime_ms"))

    overall_batch_keys = [
        "batch_input_write_overall_ms",
        "batch_kernel_overall_ms",
        "batch_read_output_overall_ms"
    ]
    for key in overall_batch_keys:
        original_overall_ms_values = timings.get(key)
        original_overall_ms_values=sorted(original_overall_ms_values)

        if not(original_overall_ms_values and len(original_overall_ms_values) == actual_batches):
            print(f"Warning: Could not find or mismatched length for {key} during incremental transformation. Skipping.")
            continue

        incremental_overall_ms_values = [0.0] * actual_batches
        incremental_overall_ms_values[-1] = original_overall_ms_values[-1]
        for i in range(1, actual_batches):
            incremental_overall_ms_values[i] = original_overall_ms_values[i] - original_overall_ms_values[i-1]
        timings[key] = incremental_overall_ms_values

    return timings


def ns_to_ms(ns_val):
    return ns_val / 1_000_000.0


def plot_timings(all_run_data):
    """
    Plots the overall C++ processing time and total batch operation times 
    against the number of batches for multiple runs.

    Args:
        all_run_data (list of dict): A list where each dict contains 
                                     the parsed timings for a single run.
    """
    if not all_run_data:
        print("No data to plot.")
        return

    batch_counts = [data.get("ActualBatchCount", 0) for data in all_run_data]
    overall_cpp_processing_times_ms = [float(data.get("OverallBatchProcessingTime_ms", 0)) for data in all_run_data]
    
    # Sum of overall times for each stage across all batches for each run
    total_batch_input_write_overall_ms = [sum(data.get("batch_input_write_overall_ms", [])) for data in all_run_data]
    total_batch_kernel_overall_ms = [sum(data.get("batch_kernel_overall_ms", [])) for data in all_run_data]
    total_batch_read_output_overall_ms = [sum(data.get("batch_read_output_overall_ms", [])) for data in all_run_data]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    ax.plot(batch_counts, overall_cpp_processing_times_ms, marker='o', linestyle='-', label='Overall C++ Processing Time')
    ax.plot(batch_counts, total_batch_input_write_overall_ms, marker='s', linestyle='--', label='Total Batch Input Write Overall Time')
    ax.plot(batch_counts, total_batch_kernel_overall_ms, marker='^', linestyle='--', label='Total Batch Kernel Overall Time')
    ax.plot(batch_counts, total_batch_read_output_overall_ms, marker='x', linestyle='--', label='Total Batch Read Output Overall Time')

    ax.set_xlabel('Number of Batches')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Performance Timings vs. Batch Size', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


def main(executable_name: str = "conv_2d_faster_analysis1.exe"):
    # Define batch sizes: start small, then increase more rapidly towards max
    batch_sizes = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450]
    max_batch_size = 450 # Defined by user
    
    # Ensure batch_sizes are sorted and unique, and do not exceed max_batch_size
    batch_sizes = sorted(list(set(bs for bs in batch_sizes if bs <= max_batch_size)))
    if not batch_sizes or batch_sizes[-1] < max_batch_size:
        if max_batch_size not in batch_sizes:
             batch_sizes.append(max_batch_size) # Ensure max_batch_size is included
    batch_sizes = sorted(list(set(batch_sizes))) # Re-sort and unique after potential append


    all_run_data = []

    executable_path = os.path.join(WORKSPACE_ROOT, "bin", executable_name)

    for batch_count_arg in batch_sizes:
        print(f"\\n--- Running C++ Analysis for {executable_name} with Batch Count: {batch_count_arg} ---")
        cpp_output = run_cpp_analysis(executable_path, batch_count_arg, WORKSPACE_ROOT)
        parsed_timings = parse_output(cpp_output)
        all_run_data.append(parsed_timings)

        print(f"  Actual Batches by C++: {parsed_timings.get('ActualBatchCount', 0)}")
        timer_res = parsed_timings.get("DeviceProfilingTimerResolution_ns", "N/A")
        print(f"  Device Profiling Timer Resolution: {timer_res} ns")

        filter_write_queue_ns = int(parsed_timings.get("FilterWriteQueueTime_ns", 0))
        filter_write_exec_ns = int(parsed_timings.get("FilterWriteExecTime_ns", 0))
        print(f"  Filter Write (Queue): {ns_to_ms(filter_write_queue_ns):.3f} ms ({filter_write_queue_ns} ns)")
        print(f"  Filter Write (Exec): {ns_to_ms(filter_write_exec_ns):.3f} ms ({filter_write_exec_ns} ns)")

        # Print sum of overall times for this run
        total_input_overall = sum(parsed_timings.get("batch_input_write_overall_ms", []))
        total_kernel_overall = sum(parsed_timings.get("batch_kernel_overall_ms", []))
        total_read_overall = sum(parsed_timings.get("batch_read_output_overall_ms", []))
        print(f"  Sum of Batch Input Write Overall: {total_input_overall:.3f} ms")
        print(f"  Sum of Batch Kernel Overall: {total_kernel_overall:.3f} ms")
        print(f"  Sum of Batch Read Output Overall: {total_read_overall:.3f} ms")

        overall_ms = float(parsed_timings.get("OverallBatchProcessingTime_ms", 0))
        print(f"  Overall C++ Processing: {overall_ms:.3f} ms")
        print("-----------------------------------------------------")

    plot_timings(all_run_data)


if __name__ == "__main__":
    main('conv_2d_faster_analysis2.exe') # Runs with default executable conv_2d_faster_analysis1.exe
    # To run with a different executable, for example "conv_2d_faster_HTP.exe":
    # main(executable_name="conv_2d_faster_HTP.exe")
