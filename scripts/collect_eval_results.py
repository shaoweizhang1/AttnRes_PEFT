import argparse
import csv
import glob
import json
import os
import re


MAIN_FIELDS = [
    "task",
    "method",
    "experiment",
    "split",
    "accuracy",
    "avg_latency",
    "avg_generated_tokens",
    "avg_tokens_per_second",
    "avg_peak_memory_mb",
    "num_examples",
    "source_path",
]


ABLATION_FIELDS = [
    "task",
    "lookback",
    "setting",
    "split",
    "accuracy",
    "avg_latency",
    "avg_generated_tokens",
    "avg_tokens_per_second",
    "avg_peak_memory_mb",
    "num_examples",
    "source_path",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_main_row(rel_path, summary):
    parts = rel_path.split(os.sep)
    experiment = parts[1]
    task = parts[2]
    split = parts[3].replace("_summary.json", "")

    suffix = f"_{task}"
    method = experiment[: -len(suffix)] if experiment.endswith(suffix) else experiment

    return {
        "task": task,
        "method": method,
        "experiment": experiment,
        "split": split,
        "accuracy": summary.get("accuracy"),
        "avg_latency": summary.get("avg_latency"),
        "avg_generated_tokens": summary.get("avg_generated_tokens"),
        "avg_tokens_per_second": summary.get("avg_tokens_per_second"),
        "avg_peak_memory_mb": summary.get("avg_peak_memory_mb"),
        "num_examples": summary.get("num_examples"),
        "source_path": rel_path,
    }


def parse_ablation_row(rel_path, summary):
    parts = rel_path.split(os.sep)
    setting = parts[1]
    task = parts[2]
    split = parts[3].replace("_summary.json", "")

    match = re.search(r"lookback_(\d+)", setting)
    lookback = int(match.group(1)) if match else None

    return {
        "task": task,
        "lookback": lookback,
        "setting": setting,
        "split": split,
        "accuracy": summary.get("accuracy"),
        "avg_latency": summary.get("avg_latency"),
        "avg_generated_tokens": summary.get("avg_generated_tokens"),
        "avg_tokens_per_second": summary.get("avg_tokens_per_second"),
        "avg_peak_memory_mb": summary.get("avg_peak_memory_mb"),
        "num_examples": summary.get("num_examples"),
        "source_path": rel_path,
    }


def collect_rows(results_dir):
    pattern = os.path.join(results_dir, "**", "*_summary.json")
    summary_paths = sorted(glob.glob(pattern, recursive=True))

    main_rows = []
    ablation_rows = []

    for path in summary_paths:
        rel_path = os.path.relpath(path, results_dir)
        summary = load_json(path)

        if rel_path.startswith(f"main{os.sep}"):
            main_rows.append(parse_main_row(rel_path, summary))
        elif rel_path.startswith(f"ablation{os.sep}"):
            ablation_rows.append(parse_ablation_row(rel_path, summary))

    main_rows.sort(key=lambda row: (row["task"], row["method"]))
    ablation_rows.sort(key=lambda row: (row["task"], row["lookback"] if row["lookback"] is not None else 10**9))
    return main_rows, ablation_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--main_output", default=None)
    parser.add_argument("--ablation_output", default=None)
    args = parser.parse_args()

    main_rows, ablation_rows = collect_rows(args.results_dir)

    main_output = args.main_output or os.path.join(args.results_dir, "main_results.csv")
    ablation_output = args.ablation_output or os.path.join(args.results_dir, "ablation_results.csv")

    write_csv(main_output, main_rows, MAIN_FIELDS)
    write_csv(ablation_output, ablation_rows, ABLATION_FIELDS)

    print(f"Saved main results to {main_output}")
    print(f"Saved ablation results to {ablation_output}")


if __name__ == "__main__":
    main()
