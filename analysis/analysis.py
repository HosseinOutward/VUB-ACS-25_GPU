# filepath: d:\User\App Files\Projects\VUB-ACS-25_GPU\analysis\analysis.py
import subprocess
import os

def run_and_parse_conv_gpu(args_list, exe_filename = "conv_2d_gpu.exe"):
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

    output_lines = completed_process.stdout.splitlines()[6:]

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
        key = key.replace('_'+str(batch_no), '')

        if cleaned_data.get(key) is None:
            cleaned_data[key] = [0]*all_batch_counts
        cleaned_data[key][batch_no-1] = value

    return cleaned_data


if __name__ == "__main__":
    output_lines_gpu = run_and_parse_conv_gpu([
            "input_lena.png",
            "kernel_edge_3c.txt",
            "true",
            "2",
            "output_gpu.png"
        ],
        exe_filename = "conv_2d_gpu.exe")
    parsed_data = clean_parsed(output_lines_gpu)

    get_cpu_time = run_and_parse_conv_gpu([
            "input_lena.png",
            "kernel_edge_3c.txt",
            "true",
            "2",
            "output.png"
        ],
        exe_filename = "conv_2d_sequential.exe")

