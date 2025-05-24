# filepath: d:\User\App Files\Projects\VUB-ACS-25_GPU\analysis\analysis.py
import subprocess
import os

# change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_and_parse_conv(args_list, exe_filename="conv_2d_gpu.exe"):
    exe_relative_path = os.path.join("..", "bin", exe_filename)
    completed_process = subprocess.run(
        [exe_relative_path] + args_list,
        capture_output=True,
        text=True,
        check=False,
        cwd=os.path.dirname(os.path.abspath(os.path.join("..", "bin")))
    )

    if completed_process.returncode != 0:
        raise RuntimeError(f"Error running {exe_filename}: {completed_process.stderr}")

    # find the first line that starts with "Overall_time"
    output_lines=None
    for i, line in enumerate(completed_process.stdout.splitlines()):
        if line.startswith("Overall_time: "):
            output_lines = completed_process.stdout.splitlines()[i:]
            break

    if output_lines[-1].find(':') == -1:
        output_lines = output_lines[:-1]

    return output_lines


def clean_parsed(output_lines):
    parsed_data = {
        parts[0].strip(): float(parts[1].strip())
        for line in output_lines
        if line.strip() and (parts := line.split(':', 1))
    }

    assert len(parsed_data) == len(output_lines), \
        f"Expected {len(output_lines)} key-value pairs, but got {len(parsed_data)}."

    cleaned_data = {'overall_time': parsed_data.pop('Overall_time')}

    if len(parsed_data) == 0:
        return cleaned_data

    log_numbers = 0
    pd_keys = list(parsed_data.keys())
    while pd_keys[log_numbers].find('_0') != -1:
        log_numbers += 1
    all_batch_counts = int(len(parsed_data) / log_numbers)

    for key, value in parsed_data.items():
        batch_no = int(key.split('_')[-1])
        key = key.replace('_' + str(batch_no), '')

        if cleaned_data.get(key) is None:
            cleaned_data[key] = [0] * all_batch_counts
        cleaned_data[key][batch_no - 1] = value

    return cleaned_data


def generate_data(samples, single_batch_size,
                  batch_counts, gpu_local_mem, cpu_threads,
                  stride=2,
                  kernel_arg="kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt",
                  input_arg="input_lena.png", ):
    args = [
        input_arg,  # input image
        kernel_arg,  # kernel files
        None,  # cpu: threads --- gpu: local mem
        f"{batch_counts}",  # batch counts
        "null",  # output file
        str(stride),  # stride
        f"{single_batch_size}"  # simulated single batch size
    ]

    parsed_data = {'cpu_overall_time': []}
    for i in range(samples):
        print(f"Sample {i + 1} of {samples}")
        args[2] = gpu_local_mem
        output_lines_gpu = run_and_parse_conv(args, exe_filename="conv_2d_gpu.exe")
        temp = clean_parsed(output_lines_gpu)

        for key, value in temp.items():
            if parsed_data.get(key) is None:
                parsed_data[key] = []
            parsed_data[key].append(value)

        # run the CPU version
        args[2] = str(cpu_threads)
        # get_cpu_time = run_and_parse_conv(args, exe_filename="conv_2d_sequential.exe")
        # parsed_data['cpu_overall_time'] += [clean_parsed(get_cpu_time)['overall_time']]

    # save the parsed data to a file
    with open("../data/out/parsed_data.txt", "w") as f:
        f.write("{")
        for key, value in parsed_data.items():
            f.write("'" + key + f"': {value},\n")
        f.write("}\n")


if __name__ == "__main__":
    kernel_x_y_fc = (5, 5, 5) # x y <channels> fc
    input_x_y_c = (1000,1000,3)
    generate_data(samples=6, single_batch_size=12, batch_counts=15,
                  gpu_local_mem='true', cpu_threads=0,
                  stride=2,
                  kernel_arg="%d,%d,%d" % kernel_x_y_fc,
                  input_arg="%d,%d,%d" % input_x_y_c,)
                  # kernel_arg="kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt",
                  # input_arg="input_lena.png")
