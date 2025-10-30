import subprocess
import numpy as np


def calculate_conv2d_ops(input_x_y_c: list, kernel_x_y_fc: list):
    """
    Calculate the number of compute and memory operations for a 2D convolution.

    Args:
        input_x_y_c: [height, width, channels] of input
        kernel_x_y_fc: [height, width, output_channels] of kernel

    Returns:
        tuple: (total_compute_ops, total_memory_ops) per output pixel
    """
    C = input_x_y_c[2]   # channel_count (input channels)
    Fw = kernel_x_y_fc[0] # filt_w
    Fh = kernel_x_y_fc[1] # filt_h
    Fc = kernel_x_y_fc[2] # filt_c (output channels)

    # Compute operations per output pixel:
    # For each output pixel, we do C * Fh * Fw multiply-accumulate operations
    # Each MAC = 1 multiply + 1 add = 2 FLOPs
    # Plus normalization operations (divide by 255, multiply by 255, clamp)
    compute_ops_per_pixel = C * Fh * Fw * 2 + 3  # 2 FLOPs per MAC + 3 for normalization

    # Memory operations per output pixel:
    # 1. Read input: C * Fh * Fw reads (one per input element in the receptive field)
    # 2. Read kernel: C * Fh * Fw reads (one per kernel weight)
    # 3. Write output: 1 write
    memory_ops_per_pixel = C * Fh * Fw + C * Fh * Fw + 1

    return compute_ops_per_pixel, memory_ops_per_pixel


def parse_device_info(exe_path):
    """
    Parse device information from executable output.

    Args:
        exe_path: Path to the device_info executable

    Returns:
        dict: Device information key-value pairs
    """
    completed_process = subprocess.run(
        [exe_path],
        capture_output=True,
        text=True,
        check=False
    )

    data = {}
    for line in completed_process.stdout.splitlines():
        if ':' in line:
            key, val = line.split(':', 1)
            data[key.strip()] = val.strip()
    return data


def theoretical_info(input_x_y_c, kernel_x_y_fc, device_info, stride=2,
                    single_batch_size=8, batch_counts=15):
    """
    Calculate theoretical performance metrics for convolution operation.

    Args:
        input_x_y_c: [height, width, channels] of input
        kernel_x_y_fc: [height, width, output_channels] of kernel
        device_info: Device information dictionary
        stride: Convolution stride (default: 2)
        single_batch_size: Size of single batch (default: 8)
        batch_counts: Number of batches (default: 15)

    Returns:
        dict: Theoretical timing and performance metrics
    """
    operation_per_pixel, memory_ops_per_pixel = calculate_conv2d_ops(input_x_y_c, kernel_x_y_fc)

    # Calculate output spatial dimensions (VALID padding)
    H_in, W_in, C_in = input_x_y_c
    K_h, K_w, C_out = kernel_x_y_fc
    H_out = (H_in - K_h) // stride + 1
    W_out = (W_in - K_w) // stride + 1
    out_size = H_out * W_out * C_out
    total_output_pixel_count = (single_batch_size * out_size) * batch_counts

    # Compute time calculation
    compute_units_needed = total_output_pixel_count
    rounds_of_parallel_compute = np.ceil(compute_units_needed / float(device_info['Estimated total cores']))
    time_per_pixel = operation_per_pixel / device_info['boost clock per core']
    gpu_compute_time = rounds_of_parallel_compute * time_per_pixel * 1000  # ms
    compute_done = operation_per_pixel * total_output_pixel_count

    # Memory time calculation
    # bytes_moved = total_output_pixel_count * memory_ops_per_pixel * 2
    bytes_moved = total_output_pixel_count +\
                  memory_ops_per_pixel * np.prod(input_x_y_c) / calculate_overlap_ratio(kernel_x_y_fc, stride)
    gpu_memory_time = bytes_moved / device_info['byte_move_speed'] * 1000  # ms

    return {
        "Output feature map size": out_size * 2,  # bytes
        "Ideal DRAM traffic": bytes_moved / 1e6,  # MB
        "Arithmetic intensity": compute_done / bytes_moved,  # FLOPs/byte
        "Time-spent compute": gpu_compute_time,  # ms
        "Time-spent memory": gpu_memory_time,  # ms
        "Time-spent overall": gpu_compute_time + gpu_memory_time,  # ms
    }


def initialize_device_info(device_info_path='../bin/device_info'):
    """
    Initialize device information with calculated parameters.

    Args:
        device_info_path: Path to device_info executable

    Returns:
        dict: Enhanced device information with calculated parameters
    """
    device_info = parse_device_info(device_info_path)

    # Add calculated parameters
    device_info['boost clock per core'] = int(
        float(device_info['Max clock frequency'][:-3]) * 1e6 * 2
    )
    device_info['byte_move_speed'] = 504.1 * 1e9  # bytes/second
    device_info['local_byte_move_speed'] = 40 * 1e9  # bytes/second

    return device_info


def calculate_overlap_ratio(kernel_size, stride):
    overlap_area = max(0, kernel_size[0] - stride) * max(0, kernel_size[1] - stride)
    total_receptive_field = kernel_size[0] * kernel_size[1]

    overlap_ratio = overlap_area / total_receptive_field

    return overlap_ratio
