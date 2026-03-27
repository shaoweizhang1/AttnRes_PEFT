"""Download and convert task datasets into Alpaca-style JSON files.

Supported tasks:
- GSM8K
- RTE
- BoolQ

Output format:
[
  {
    "instruction": "...",
    "input": "...",
    "output": "...",
    "task": "...",
    "split": "..."
  }
]
"""
import os

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from huggingface_hub import snapshot_download


MODEL_NAME = "Qwen/Qwen2.5-7B"
SAVE_DIR = "./model"
DEFAULT_OUT_DIR = Path("data")


def _write_json(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_gsm8k(example, split):
    answer = example["answer"].strip()
    final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
    return {
        "instruction": "Solve the math word problem and give the final answer.",
        "input": example["question"].strip(),
        "output": final_answer,
        "task": "gsm8k",
        "split": split,
    }


def format_rte(example, split):
    label_map = {0: "entailment", 1: "not_entailment"}
    return {
        "instruction": (
            "Determine whether the hypothesis is entailed by the premise. "
            "Answer with entailment or not_entailment."
        ),
        "input": (
            f"Premise: {example['sentence1'].strip()}\n"
            f"Hypothesis: {example['sentence2'].strip()}"
        ),
        "output": label_map[int(example["label"])],
        "task": "rte",
        "split": split,
    }


def format_boolq(example, split):
    label = "yes" if bool(example["label"]) else "no"
    return {
        "instruction": "Read the passage and answer the question with yes or no.",
        "input": (
            f"Passage: {example['passage'].strip()}\n"
            f"Question: {example['question'].strip()}"
        ),
        "output": label,
        "task": "boolq",
        "split": split,
    }


class DataLoader:
    def __init__(self, output_dir=DEFAULT_OUT_DIR):
        self.output_dir = Path(output_dir)

    def process_gsm8k(self):
        dataset = load_dataset("gsm8k", "main")
        for split in ("train", "test"):
            records = []
            for example in tqdm(dataset[split], desc=f"Processing gsm8k {split}"):
                records.append(format_gsm8k(example, split))
            _write_json(records, self.output_dir / "gsm8k" / f"{split}.json")

    def process_rte(self):
        dataset = load_dataset("glue", "rte")
        for split in ("train", "validation"):
            records = []
            for example in tqdm(dataset[split], desc=f"Processing rte {split}"):
                records.append(format_rte(example, split))
            _write_json(records, self.output_dir / "rte" / f"{split}.json")

    def process_boolq(self):
        dataset = load_dataset("super_glue", "boolq")
        for split in ("train", "validation"):
            records = []
            for example in tqdm(dataset[split], desc=f"Processing boolq {split}"):
                records.append(format_boolq(example, split))
            _write_json(records, self.output_dir / "boolq" / f"{split}.json")

    def run(self):
        self.process_gsm8k()
        self.process_rte()
        self.process_boolq()



class ModelLoader:
    def __init__(self, model_name=MODEL_NAME, save_dir=SAVE_DIR):
        self.model_name = model_name
        self.save_dir = save_dir

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)

        snapshot_download(
            repo_id=self.model_name,
            local_dir=self.save_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
