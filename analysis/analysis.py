import subprocess
import re
import matplotlib.pyplot as plt
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SCRIPT_DIR)
EXECUTABLE_PATH = os.path.join(WORKSPACE_ROOT, "bin", "conv_2d_faster_analysis1.exe")

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
            if "_ns" in key or "Resolution_ns" in key: 
                timings[key] = int(value) if value.isdigit() else value
            elif "_ms" in key:
                timings[key] = float(value)
            elif "ActualBatchCount" in key:
                timings["ActualBatchCount"] = int(value)

    actual_batches = timings.get("ActualBatchCount", 0)
    # Initialize lists for both queue and exec times for each batch operation type
    timings["batch_input_write_queue_ns"] = [0] * actual_batches
    timings["batch_input_write_exec_ns"] = [0] * actual_batches
    timings["batch_kernel_queue_ns"] = [0] * actual_batches
    timings["batch_kernel_exec_ns"] = [0] * actual_batches
    timings["batch_read_output_queue_ns"] = [0] * actual_batches
    timings["batch_read_output_exec_ns"] = [0] * actual_batches

    for i in range(actual_batches):
        timings["batch_input_write_queue_ns"][i] = timings.get(f"Batch_{i}_InputWriteQueueTime_ns")
        timings["batch_input_write_exec_ns"][i] = timings.get(f"Batch_{i}_InputWriteExecTime_ns")
        timings["batch_kernel_queue_ns"][i] = timings.get(f"Batch_{i}_KernelQueueTime_ns")
        timings["batch_kernel_exec_ns"][i] = timings.get(f"Batch_{i}_KernelExecTime_ns")
        timings["batch_read_output_queue_ns"][i] = timings.get(f"Batch_{i}_ReadQueueTime_ns")
        timings["batch_read_output_exec_ns"][i] = timings.get(f"Batch_{i}_ReadExecTime_ns")
    
    return timings

def ns_to_ms(ns_val):
    return ns_val / 1_000_000.0

def plot_timings(timings_data): # Removed tested_batch_count argument
    actual_batch_count = timings_data.get("ActualBatchCount", 0)
    
    y_category_definitions = [] 

    # 1. Overall Processing
    overall_time_ms = timings_data.get("OverallBatchProcessingTime_ms", 0)
    if overall_time_ms > 0:
        y_category_definitions.append(("Overall Processing", 'single', overall_time_ms, 'purple'))

    # Filter Write Queue Time
    filter_write_queue_ms = ns_to_ms(timings_data.get("FilterWriteQueueTime_ns", 0))
    if filter_write_queue_ms > 0:
        y_category_definitions.append(("Filter Write (Queue)", 'single', filter_write_queue_ms, 'lightskyblue'))
    # Filter Write Exec Time
    filter_write_exec_ms = ns_to_ms(timings_data.get("FilterWriteExecTime_ns", 0))
    if filter_write_exec_ms > 0:
        y_category_definitions.append(("Filter Write (Exec)", 'single', filter_write_exec_ms, 'deepskyblue'))

    # Batch data sources for Queue times
    batch_input_writes_queue_ms = [ns_to_ms(t) for t in timings_data.get("batch_input_write_queue_ns", [])]
    batch_kernel_execs_queue_ms = [ns_to_ms(t) for t in timings_data.get("batch_kernel_queue_ns", [])]
    batch_read_outputs_queue_ms = [ns_to_ms(t) for t in timings_data.get("batch_read_output_queue_ns", [])]

    # Batch data sources for Exec times
    batch_input_writes_exec_ms = [ns_to_ms(t) for t in timings_data.get("batch_input_write_exec_ns", [])]
    batch_kernel_execs_exec_ms = [ns_to_ms(t) for t in timings_data.get("batch_kernel_exec_ns", [])]
    batch_read_outputs_exec_ms = [ns_to_ms(t) for t in timings_data.get("batch_read_output_exec_ns", [])]

    # Batch colors - using tab10 colormap, cycles if more than 10 batches
    batch_color_palette = [plt.cm.tab10(i % 10) for i in range(actual_batch_count if actual_batch_count > 0 else 1)]

    # Input Write Queue Times
    if any(t > 0 for t in batch_input_writes_queue_ms):
        y_category_definitions.append(("Input Write (Queue)", 'stacked', batch_input_writes_queue_ms, batch_color_palette))
    # Input Write Exec Times
    if any(t > 0 for t in batch_input_writes_exec_ms):
        y_category_definitions.append(("Input Write (Exec)", 'stacked', batch_input_writes_exec_ms, batch_color_palette))
    
    # Kernel Exec Queue Times
    if any(t > 0 for t in batch_kernel_execs_queue_ms):
        y_category_definitions.append(("Kernel (Queue)", 'stacked', batch_kernel_execs_queue_ms, batch_color_palette))
    # Kernel Exec Exec Times
    if any(t > 0 for t in batch_kernel_execs_exec_ms):
        y_category_definitions.append(("Kernel (Exec)", 'stacked', batch_kernel_execs_exec_ms, batch_color_palette))

    # Read Output Queue Times
    if any(t > 0 for t in batch_read_outputs_queue_ms):
        y_category_definitions.append(("Read Output (Queue)", 'stacked', batch_read_outputs_queue_ms, batch_color_palette))
    # Read Output Exec Times
    if any(t > 0 for t in batch_read_outputs_exec_ms):
        y_category_definitions.append(("Read Output (Exec)", 'stacked', batch_read_outputs_exec_ms, batch_color_palette))
    
    if not y_category_definitions:
        print("No timing data > 0 ms to plot.")
        return

    num_categories = len(y_category_definitions)
    fig, ax = plt.subplots(figsize=(20, num_categories * 0.7 + 2), dpi=200)
    
    y_pos = np.arange(num_categories)
    category_labels = [item[0] for item in y_category_definitions]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(category_labels)
    ax.invert_yaxis() 
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'Performance Timings (Actual Batches: {actual_batch_count})', fontsize=14)

    max_total_time_on_x_axis = 0 
    plotted_batch_legend_colors = {} 

    for i, (cat_name, type, data_values, colors_for_cat) in enumerate(y_category_definitions):
        current_y_pos = y_pos[i]
        if type == 'single':
            val_ms = data_values 
            bar_color = colors_for_cat 
            ax.barh(current_y_pos, val_ms, color=bar_color, align='center', height=0.6)
            if val_ms > 0:
                 ax.text(val_ms * 1.01, current_y_pos, f'{val_ms:.3f} ms', ha='left', va='center', fontsize=9)
            if val_ms > max_total_time_on_x_axis: max_total_time_on_x_axis = val_ms
        
        elif type == 'stacked':
            left_offset = 0
            # Filter out zero values before summing for total, to avoid issues if all are zero
            positive_data_values = [v for v in data_values if v > 0]
            total_stacked_time = sum(positive_data_values) if positive_data_values else 0

            for batch_idx, val_ms in enumerate(data_values):
                if val_ms > 0:
                    # Ensure batch_idx is within the range of the color palette
                    batch_color = colors_for_cat[batch_idx % len(colors_for_cat)]
                    ax.barh(current_y_pos, val_ms, left=left_offset, color=batch_color, align='center', height=0.6)
                    
                    if batch_idx not in plotted_batch_legend_colors:
                        plotted_batch_legend_colors[batch_idx] = batch_color
                    
                    # Determine text color for segment based on background luminance
                    r, g, b, _ = batch_color # RGBA
                    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                    text_color = 'white' if luminance < 0.5 else 'black'
                    
                    # Only add text if segment is wide enough
                    if val_ms / (total_stacked_time if total_stacked_time > 0 else 1) > 0.05 : # If segment is >5% of total
                        ax.text(left_offset + val_ms / 2, current_y_pos, f'{val_ms:.3f}', 
                                ha='center', va='center', fontsize=8, color=text_color, fontweight='medium')
                left_offset += val_ms
            
            if total_stacked_time > 0: 
                 ax.text(total_stacked_time * 1.01, current_y_pos, f'Total: {total_stacked_time:.3f} ms', 
                         ha='left', va='center', fontsize=9)
            if total_stacked_time > max_total_time_on_x_axis: max_total_time_on_x_axis = total_stacked_time

    if max_total_time_on_x_axis > 0:
        ax.set_xlim(right=max_total_time_on_x_axis * 1.22) 

    if plotted_batch_legend_colors and actual_batch_count > 0:
        legend_elements = []
        for batch_idx in sorted(plotted_batch_legend_colors.keys()):
            if batch_idx < actual_batch_count: 
                color = plotted_batch_legend_colors[batch_idx]
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'Batch {batch_idx}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, title="Batches", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9, title_fontsize=10)
            # Adjust layout to make space for the legend
            plt.subplots_adjust(right=0.85) # Example: use 85% of the figure width for the plot
    
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)
    # Ensure tight_layout is called after subplots_adjust if used, or manage spacing carefully
    if not (plotted_batch_legend_colors and actual_batch_count > 0 and legend_elements):
        plt.tight_layout() # Call only if legend didn't adjust subplots

    plt.show()

def main(batch_count_arg: int = 10):
    cpp_output = run_cpp_analysis(EXECUTABLE_PATH, batch_count_arg, WORKSPACE_ROOT)
    parsed_timings = parse_output(cpp_output)

    print("\n--- Parsed Timings ---")
    actual_batches = parsed_timings.get("ActualBatchCount", 0)
    print(f"  Actual Batches by C++: {actual_batches}")
    timer_res = parsed_timings.get("DeviceProfilingTimerResolution_ns", "N/A")
    print(f"  Device Profiling Timer Resolution: {timer_res} ns")
    
    filter_write_queue_ns = parsed_timings.get("FilterWriteQueueTime_ns", 0)
    filter_write_exec_ns = parsed_timings.get("FilterWriteExecTime_ns", 0)
    print(f"  Filter Write (Queue): {ns_to_ms(filter_write_queue_ns):.3f} ms ({filter_write_queue_ns} ns)")
    print(f"  Filter Write (Exec): {ns_to_ms(filter_write_exec_ns):.3f} ms ({filter_write_exec_ns} ns)")

    for i in range(actual_batches):
        input_w_queue_ns = parsed_timings["batch_input_write_queue_ns"][i]
        input_w_exec_ns = parsed_timings["batch_input_write_exec_ns"][i]
        kernel_e_queue_ns = parsed_timings["batch_kernel_queue_ns"][i]
        kernel_e_exec_ns = parsed_timings["batch_kernel_exec_ns"][i]
        read_t_queue_ns = parsed_timings["batch_read_output_queue_ns"][i]
        read_t_exec_ns = parsed_timings["batch_read_output_exec_ns"][i]
        print(f"  Batch {i}:")
        print(f"    Input Write (Queue): {ns_to_ms(input_w_queue_ns):.3f} ms ({input_w_queue_ns} ns)")
        print(f"    Input Write (Exec): {ns_to_ms(input_w_exec_ns):.3f} ms ({input_w_exec_ns} ns)")
        print(f"    Kernel (Queue): {ns_to_ms(kernel_e_queue_ns):.3f} ms ({kernel_e_queue_ns} ns)")
        print(f"    Kernel (Exec): {ns_to_ms(kernel_e_exec_ns):.3f} ms ({kernel_e_exec_ns} ns)")
        print(f"    Read Output (Queue): {ns_to_ms(read_t_queue_ns):.3f} ms ({read_t_queue_ns} ns)")
        print(f"    Read Output (Exec): {ns_to_ms(read_t_exec_ns):.3f} ms ({read_t_exec_ns} ns)")
    
    overall_ms = parsed_timings.get("OverallBatchProcessingTime_ms", 0)
    print(f"  Overall C++ Processing: {overall_ms:.3f} ms")
    print("--------------------------\n")

    plot_timings(parsed_timings) # Removed batch_count_arg

if __name__ == "__main__":
    main(200)
