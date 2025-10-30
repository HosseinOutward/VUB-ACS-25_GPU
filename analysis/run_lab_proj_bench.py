#!/usr/bin/env python3
import argparse
import math
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None
    _matplotlib_import_error = e

# ----------------------
# Parsing and utilities
# ----------------------

HOST_RE = re.compile(r"Host time:\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
GPU_RE = re.compile(r"GPU time:\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE)
GOPS_RE = re.compile(r"Performance:\s*([0-9]+\.?[0-9]*)\s*GOPS", re.IGNORECASE)


def parse_sizes_arg(s: Optional[str], min_size: Optional[int], max_size: Optional[int], step: Optional[int]) -> List[int]:
    """Parse sizes from string or generate from min/max/step"""
    if s:
        s = s.strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 3:
                raise ValueError("--sizes must be 'start:stop:step' or a comma list")
            start, stop, st = (int(float(p)) for p in parts)
            if st <= 0:
                raise ValueError("step must be positive")
            return list(range(start, stop + (1 if st > 0 else -1), st))
        else:
            return [int(float(x)) for x in s.split(",") if x.strip()]
    if min_size and max_size and step:
        if step <= 0:
            raise ValueError("--step must be positive")
        return list(range(min_size, max_size + (1 if step > 0 else -1), step))
    # Default: modest sweep
    return [int(x) for x in [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6]]


def pick_default_executable(root: Path) -> Path:
    """Find the lab_proj executable"""
    candidates = [
        root / "bin" / "lab_proj.exe",
        root / "bin" / "lab_proj",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # Return first candidate even if not found


# ----------------------
# Runner
# ----------------------

def run_once(exe: Path, kernel: str, size: int, reps: int, ftype: int, timeout: Optional[float], elements_per_thread: int = 1) -> Tuple[Optional[float], Optional[float], Optional[float], str, str]:
    """Run lab_proj once and parse output"""
    # Check if executable exists
    if not exe.exists():
        return None, None, None, f"Executable not found: {exe}", ""

    # Try with 5 arguments first (kernel, size, reps, float_type, elements_per_thread)
    cmd = [str(exe), kernel, str(size), str(reps), str(ftype), str(elements_per_thread)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

        # If it fails with usage error, try with 4 arguments (without elements_per_thread)
        if res.returncode != 0 and "Usage:" in (res.stdout or ""):
            cmd = [str(exe), kernel, str(size), str(reps), str(ftype)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

            # If still fails, try with 3 arguments (legacy mode)
            if res.returncode != 0 and "Usage:" in (res.stdout or ""):
                cmd = [str(exe), kernel, str(size), str(reps)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)

    except subprocess.TimeoutExpired:
        return None, None, None, f"TIMEOUT after {timeout}s: {' '.join(shlex.quote(c) for c in cmd)}", ""

    out = (res.stdout or "").strip()
    err = (res.stderr or "").strip()

    if res.returncode != 0:
        return None, None, None, f"RETURN CODE {res.returncode}:\nSTDOUT:\n{out}\nSTDERR:\n{err}", out

    host_ms = None
    gpu_ms = None
    gops = None

    # Parse output using regex
    m = HOST_RE.search(out)
    if m:
        host_ms = float(m.group(1))
    m = GPU_RE.search(out)
    if m:
        gpu_ms = float(m.group(1))
    m = GOPS_RE.search(out)
    if m:
        gops = float(m.group(1))

    # If GOPS missing but gpu time available, compute it
    if gops is None and gpu_ms is not None and gpu_ms > 0:
        ops = float(size) * float(reps)
        gops = ops / (gpu_ms / 1000.0) / 1e9

    return host_ms, gpu_ms, gops, "", out


# ----------------------
# Aggregation
# ----------------------

def aggregate(results: List[Dict]) -> List[Dict]:
    """Aggregate results by size"""
    agg: Dict[int, Dict[str, List[float]]] = {}
    for r in results:
        size = r["size"]
        bucket = agg.setdefault(size, {"host_ms": [], "gpu_ms": [], "gops": []})
        for k in ["host_ms", "gpu_ms", "gops"]:
            v = r.get(k)
            if v is not None and not math.isnan(v):
                bucket[k].append(float(v))

    out: List[Dict] = []
    for size, bucket in sorted(agg.items(), key=lambda kv: kv[0]):
        row: Dict[str, float] = {"size": size}
        for k in ["host_ms", "gpu_ms", "gops"]:
            vals = bucket[k]
            if vals:
                # Calculate both mean and median statistics
                row[k + "_mean"] = statistics.fmean(vals)
                row[k + "_median"] = statistics.median(vals)
                row[k + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            else:
                row[k + "_mean"] = float("nan")
                row[k + "_median"] = float("nan")
                row[k + "_std"] = float("nan")
        out.append(row)
    return out


# ----------------------
# Plotting
# ----------------------

def plot_aggregates(agg_rows: List[Dict], metric: str, title: str, out_path: Path):
    """Create a plot for a given metric"""
    if plt is None:
        raise RuntimeError(f"matplotlib is required but failed to import: {_matplotlib_import_error}")

    xs = [row["size"] for row in agg_rows]

    # Use median values if available, otherwise fall back to mean
    median_key = f"{metric}_median"
    mean_key = f"{metric}_mean"

    if median_key in agg_rows[0] and not math.isnan(agg_rows[0][median_key]):
        ys = [row[median_key] for row in agg_rows]
        stat_type = "median"
    else:
        ys = [row[mean_key] for row in agg_rows]
        stat_type = "mean"

    yerr = [row[f"{metric}_std"] for row in agg_rows]

    plt.figure(figsize=(10, 6))
    plt.errorbar(xs, ys, yerr=yerr, fmt="-o", capsize=3, linewidth=1.5, markersize=4)
    plt.title(f"{title} ({stat_type})")
    plt.xlabel("Array size (elements)")
    ylabels = {
        "gops": "Throughput (GOPS)",
        "gpu_ms": "GPU time (ms)",
        "host_ms": "Host time (ms)",
    }
    plt.ylabel(ylabels.get(metric, metric))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------
# CSV writing
# ----------------------

def write_csv(path: Path, rows: List[Dict]):
    """Write results to CSV"""
    if not rows:
        return
    keys = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k)
                if v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


# ----------------------
# Public API for Jupyter/notebooks
# ----------------------

def run_bench(
    kernel: str = "sum",
    sizes: Optional[List[int]] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    step: Optional[int] = None,
    repetitions: int = 100,
    float_type: int = 32,
    runs_per_size: int = 5,  # Changed from 3 to 5
    timeout: float = 120.0,
    exe: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    plot_metrics: Sequence[str] = ("gops",),
    raw_csv: str = "raw_results.csv",
    agg_csv: str = "aggregated_results.csv",
    make_plots: bool = True,
    verbose: bool = True,
    elements_per_thread: int = 1,
) -> Dict:
    """
    Run lab_proj across sizes and return results.

    Returns a dict with keys: results, aggregates, raw_csv, agg_csv, plots, stdout, failures, exe
    """
    repo_root = Path(__file__).resolve().parents[1]
    if exe is None:
        exe = pick_default_executable(repo_root)
    if save_dir is None:
        save_dir = repo_root / "analysis" / "bench_out"

    if sizes is None:
        sizes = parse_sizes_arg(None, min_size, max_size, step)

    results: List[Dict] = []
    failures: List[str] = []
    stdout_blobs: List[str] = []

    if verbose:
        print(f"Running {kernel} across {len(sizes)} sizes with reps={repetitions}, float{float_type}, elements_per_thread={elements_per_thread} using {exe}")

    for size in sizes:
        for run_idx in range(runs_per_size):
            host_ms, gpu_ms, gops, err, out_txt = run_once(exe, kernel, size, repetitions, float_type, timeout, elements_per_thread)
            stdout_blobs.append(out_txt)
            if err:
                failures.append(f"size={size} run={run_idx+1}: {err}")
                if verbose:
                    print(f"[FAIL] size={size} run={run_idx+1}: {err}")
                continue

            results.append({
                "size": size,
                "run": run_idx + 1,
                "host_ms": host_ms,
                "gpu_ms": gpu_ms,
                "gops": gops,
                "elements_per_thread": elements_per_thread,
            })

            if verbose:
                print(f"[OK] size={size} run={run_idx+1}: host_ms={host_ms} gpu_ms={gpu_ms} gops={gops}")

    # Save CSV files
    raw_csv_path = save_dir / raw_csv
    write_csv(raw_csv_path, results)

    agg = aggregate(results)
    agg_csv_path = save_dir / agg_csv
    write_csv(agg_csv_path, agg)

    # Create plots
    plot_paths: Dict[str, Path] = {}
    if make_plots:
        metrics = list(plot_metrics)
        for metric in metrics:
            out_png = save_dir / f"plot_{metric}_{kernel}_f{float_type}_r{repetitions}_ept{elements_per_thread}.png"
            title = f"lab_proj {kernel} float{float_type} reps={repetitions} ept={elements_per_thread} — {metric}"
            try:
                plot_aggregates(agg, metric, title, out_png)
                plot_paths[metric] = out_png
                if verbose:
                    print(f"Saved plot: {out_png}")
            except Exception as e:
                failures.append(f"Plotting failed for metric {metric}: {e}")

    return {
        "results": results,
        "aggregates": agg,
        "raw_csv": raw_csv_path,
        "agg_csv": agg_csv_path,
        "plots": plot_paths,
        "stdout": stdout_blobs,
        "failures": failures,
        "exe": exe,
    }


# ----------------------
# Main CLI
# ----------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_exe = pick_default_executable(repo_root)

    p = argparse.ArgumentParser(description="Run lab_proj benchmarks across array sizes and plot results.")
    p.add_argument("--exe", type=Path, default=default_exe, help="Path to lab_proj executable")
    p.add_argument("--kernel", choices=["sum", "fma"], default="sum", help="Kernel type to run")
    p.add_argument("--sizes", type=str, default=None, help="Sizes as 'start:stop:step' or comma list")
    p.add_argument("--min-size", type=int, default=None, help="Minimum size if --sizes not given")
    p.add_argument("--max-size", type=int, default=None, help="Maximum size if --sizes not given")
    p.add_argument("--step", type=int, default=None, help="Step if --sizes not given")
    p.add_argument("--repetitions", type=int, default=100, help="Compute repetitions per element")
    p.add_argument("--float-type", type=int, choices=[16, 32], default=32, help="Data precision: 16 or 32")
    p.add_argument("--runs-per-size", type=int, default=5, help="Repeat each size this many times")  # Changed from 3 to 5
    p.add_argument("--timeout", type=float, default=120.0, help="Per-run timeout in seconds")
    p.add_argument("--save-dir", type=Path, default=repo_root / "analysis" / "bench_out", help="Directory to save outputs")
    p.add_argument("--plot-metrics", type=str, default="gops", help="Comma list: gops,gpu_ms,host_ms")
    p.add_argument("--raw-csv", type=str, default="raw_results.csv", help="Raw results filename")
    p.add_argument("--agg-csv", type=str, default="aggregated_results.csv", help="Aggregated results filename")
    p.add_argument("--elements-per-thread", type=int, default=1, help="Number of elements processed per thread")

    args = p.parse_args(argv)

    exe = args.exe
    if not exe.exists():
        print(f"Warning: executable not found at {exe}. Build the project first.", file=sys.stderr)

    result = run_bench(
        kernel=args.kernel,
        sizes=parse_sizes_arg(args.sizes, args.min_size, args.max_size, args.step),
        repetitions=args.repetitions,
        float_type=args.float_type,
        runs_per_size=args.runs_per_size,
        timeout=args.timeout,
        exe=exe,
        save_dir=args.save_dir,
        plot_metrics=[m.strip() for m in args.plot_metrics.split(",") if m.strip()],
        raw_csv=args.raw_csv,
        agg_csv=args.agg_csv,
        make_plots=True,
        verbose=True,
        elements_per_thread=args.elements_per_thread,
    )

    if result["failures"]:
        print(f"\n{len(result['failures'])} failures occurred:", file=sys.stderr)
        for f in result["failures"][:5]:
            print(f" - {f}", file=sys.stderr)
        if len(result["failures"]) > 5:
            print(f" ... and {len(result['failures'])-5} more", file=sys.stderr)

    print(f"\nResults saved to:")
    print(f"Raw CSV: {result['raw_csv']}")
    print(f"Aggregated CSV: {result['agg_csv']}")
    for metric, path in result["plots"].items():
        print(f"Plot ({metric}): {path}")

    return 0 if result["results"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
