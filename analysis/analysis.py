# filepath: d:\User\App Files\Projects\VUB-ACS-25_GPU\analysis\analysis.py
import subprocess
import os


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
        print(f"Error: {completed_process.stderr}")

        return None

    # find the first line that starts with "Overall_time"
    for i, line in enumerate(completed_process.stdout.splitlines()):
        if line.startswith("Overall_time: "): break
    output_lines = completed_process.stdout.splitlines()[i:]

    if output_lines[-1].find(':') == -1:
        output_lines = output_lines[:-1]

    return output_lines


def clean_parsed(output_lines):
    parsed_data = {
        parts[0].strip(): int(parts[1].strip())
        for line in output_lines
        if line.strip() and (parts := line.split(':', 1))
    }

    assert len(parsed_data) == len(output_lines), \
        f"Expected {len(output_lines)} key-value pairs, but got {len(parsed_data)}."

    cleaned_data = {'overall_time': parsed_data.pop('Overall_time')}
    all_batch_counts = int(len(parsed_data) / 4)

    for key, value in parsed_data.items():
        batch_no = int(key.split('_')[-1])
        key = key.replace('_' + str(batch_no), '')

        if cleaned_data.get(key) is None:
            cleaned_data[key] = [0] * all_batch_counts
        cleaned_data[key][batch_no - 1] = value

    return cleaned_data

def generate_data(samples,
        single_batch_size = 3, batch_counts = 2,
        gpu_local_mem = 'false', cpu_threads = '0'):
    args = [
        "input_lena.png",  # input image
        "kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt",  # kernel files
        None,  # cpu: threads --- gpu: local mem
        f"{batch_counts}",  # batch counts
        "null",  # output file
        "2",  # stride
        f"{single_batch_size}"  # simulated single batch size
    ]

    parsed_data = {'cpu_overall_time':[]}
    for _ in range(samples):
        args[2] = gpu_local_mem
        output_lines_gpu = run_and_parse_conv(args, exe_filename="conv_2d_gpu.exe")
        temp = clean_parsed(output_lines_gpu)

        for key, value in temp.items():
            if parsed_data.get(key) is None:
                parsed_data[key] = []
            parsed_data[key].append(value)

        # run the CPU version
        args[2] = cpu_threads
        get_cpu_time = run_and_parse_conv(args, exe_filename="conv_2d_sequential.exe")
        parsed_data['cpu_overall_time'] += [clean_parsed(get_cpu_time)['overall_time']]

    # save the parsed data to a file
    with open("../data/out/parsed_data.txt", "w") as f:
        f.write("{")
        for key, value in parsed_data.items():
            f.write("'"+key+f"': {value},\n")
        f.write("}\n")

if __name__ == "__main__":
    generate_data(samples=100,)