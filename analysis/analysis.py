# filepath: d:\User\App Files\Projects\VUB-ACS-25_GPU\analysis\analysis.py
import subprocess
import os

import numpy as np

# change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_and_parse_conv(args_list, exe_filename="conv_2d_gpu"):
    # Resolve the correct executable across platforms and use an absolute path
    project_root = os.path.dirname(os.path.abspath(os.path.join("..", "bin")))
    bin_dir = os.path.abspath(os.path.join(project_root, "bin"))

    # Accept either with or without .exe; prefer platform-typical first
    specified = exe_filename
    base = specified[:-4] if specified.lower().endswith(".exe") else specified
    with_ext = base + ".exe"
    without_ext = base

    candidates = [with_ext, without_ext] if os.name == "nt" else [without_ext, with_ext]

    exe_path = None
    for name in candidates:
        candidate_path = os.path.join(bin_dir, name)
        if os.path.exists(candidate_path):
            exe_path = candidate_path
            break

    if exe_path is None:
        raise FileNotFoundError(
            f"Executable not found. Looked for {candidates} in {bin_dir}. "
            f"Ensure the binaries are built and located in the bin directory."
        )

    # Ensure executable bit on POSIX if missing
    if os.name != "nt" and not os.access(exe_path, os.X_OK):
        try:
            # add u+x,g+x,o+x
            os.chmod(exe_path, os.stat(exe_path).st_mode | 0o111)
        except Exception as e:
            raise PermissionError(
                f"Executable at {exe_path} is not executable and chmod failed: {e}. "
                f"Try: chmod +x '{exe_path}'"
            )

    completed_process = subprocess.run(
        [exe_path] + args_list,
        capture_output=True,
        text=True,
        check=False,
        cwd=project_root,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(f"Error running {os.path.basename(exe_path)}: {completed_process.stderr}, {completed_process.stdout}")

    # find the first line that starts with "Overall_time"
    output_lines = None
    stdout_lines = completed_process.stdout.splitlines()
    for i, line in enumerate(stdout_lines):
        if line.startswith("Overall_time: "):
            output_lines = stdout_lines[i:]
            break

    if not output_lines:
        raise ValueError(
            "Did not find 'Overall_time: ' in program output. "
            f"First 10 lines: {stdout_lines[:10]} | stderr: {completed_process.stderr[:2000]}"
        )

    if ":" not in output_lines[-1]:
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
    for key, value in list(parsed_data.items()):
        if 'gpu_overall_time_' in key:
            cleaned_data[key] = parsed_data.pop(key)

    if len(parsed_data) == 0:
        return cleaned_data

    log_numbers = 0
    pd_keys = list(parsed_data.keys())
    while log_numbers<len(pd_keys) and pd_keys[log_numbers].find('_0') != -1:
        log_numbers += 1
    all_batch_counts = int(len(parsed_data) / log_numbers)

    for key, value in parsed_data.items():
        batch_no = int(key.split('_')[-1])
        key = key.replace('_' + str(batch_no), '')

        if cleaned_data.get(key) is None:
            cleaned_data[key] = [0] * all_batch_counts
        cleaned_data[key][batch_no - 1] = value

    return cleaned_data


def generate_data(
        cpu_or_gpu, cpu_threads,
        samples, single_batch_size, batch_counts,

        kernel_arg="kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt",
        input_arg="input_lena.png", stride=2,
        gpu_detailed_timing=None,
        gpu_local_mem=None, gpu_pinned_mem=None, gpu_full_duplex=None, float16=False,
        conv2d_file_name="conv_2d_gpu"
    ):
    args = [
        input_arg,  # input image
        kernel_arg,  # kernel files
        f"{batch_counts}",  # batch counts
        "null",  # output file
        str(stride),  # stride
        f"{single_batch_size}",  # simulated single batch size
        None,  # cpu: threads --- gpu: local mem
        str(gpu_detailed_timing).lower(),  # detailed timings
        str(gpu_pinned_mem).lower(),  # pinned memory
        str(gpu_full_duplex).lower(),  # full duplex
        str(float16).lower(),  # float16
    ]

    print(f'{cpu_or_gpu=}')
    assert cpu_or_gpu in ['cpu','gpu','both']

    parsed_data = {}
    for i in range(samples):
        print(f"Sample {i + 1} of {samples}")
        if cpu_or_gpu in ['gpu', 'both']:
            assert gpu_detailed_timing in [False,True], gpu_detailed_timing
            assert gpu_local_mem in [False,True], gpu_local_mem
            assert gpu_pinned_mem in [False,True], gpu_pinned_mem
            assert gpu_full_duplex in [False,True], gpu_full_duplex
            assert float16 in [False,True], float16

            args[6] = str(cpu_threads)
            output_lines_gpu = run_and_parse_conv(args, exe_filename=conv2d_file_name)
            temp = clean_parsed(output_lines_gpu)

            for key, value in temp.items():
                if parsed_data.get(key) is None:
                    parsed_data[key] = []
                parsed_data[key].append(value)

        # run the CPU version
        if cpu_or_gpu in ['cpu', 'both']:
            if 'cpu_overall_time' not in parsed_data.keys():
                parsed_data['cpu_overall_time']=[]

            args[6] = str(cpu_threads)
            get_cpu_time = run_and_parse_conv(args, exe_filename="conv_2d_sequential")
            parsed_data['cpu_overall_time'] += [clean_parsed(get_cpu_time)['overall_time']]

    return parsed_data


if __name__ == "__main__":
    kernel_x_y_fc = (4, 4, 10) # x y <channels> fc
    input_x_y_c = (1000,1000,40)
    stride=2
    samples=3
    single_batch_size=8
    batch_counts=15

    device = 'gpu'
    gpu_detailed_timing = True
    gpu_local_mem = True; gpu_pinned_mem = True; gpu_full_duplex = True

    for gpu_local_mem, gpu_pinned_mem, gpu_full_duplex in [[0,0,0],[1,0,0]]:#np.ndindex(2, 2, 2):
        gpu_local_mem, gpu_pinned_mem, gpu_full_duplex =\
            bool(1-gpu_local_mem), bool(1-gpu_pinned_mem), bool(1-gpu_full_duplex)
        print(f"Running with gpu_local_mem={gpu_local_mem}, "
              f"gpu_pinned_mem={gpu_pinned_mem}, gpu_full_duplex={gpu_full_duplex}")
        parsed_data = generate_data(
            stride=stride,
            kernel_arg="%d,%d,%d" % kernel_x_y_fc,
            input_arg="%d,%d,%d" % input_x_y_c,
            # kernel_arg="kernel_edge_1c.txt,kernel_edge_2c.txt,kernel_edge_3c.txt",
            # input_arg="input_lena.png",

            samples=samples, single_batch_size=single_batch_size, batch_counts=batch_counts,
            cpu_or_gpu=device, cpu_threads=0,
            gpu_detailed_timing=gpu_detailed_timing,
            gpu_local_mem=gpu_local_mem, gpu_pinned_mem=gpu_pinned_mem, gpu_full_duplex=gpu_full_duplex)

        file_name = 'report_'+device+"_lm-"+str(gpu_local_mem)+"_pm-"+str(gpu_pinned_mem)+'_fd-'+str(gpu_full_duplex)
        file_name = file_name if device == 'gpu' else 'report_cpu'
        print('saved to', file_name)
        with open(f"../data/out/timings/{file_name}.txt", "w") as f:
            f.write("{")
            for key, value in parsed_data.items():
                f.write("'" + key + f"': {value},\n")
            f.write("}\n")