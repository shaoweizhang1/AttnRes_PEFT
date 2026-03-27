import argparse
import os
import re
import time

import torch
from tqdm import tqdm
from src.utils import build_prompt, get_device, load_json, load_model, load_tokenizer, save_json


DEFAULT_MODEL_DIR = "./model"
DEFAULT_SAVE_DIR = "./results"


def normalize_text(text):
    return text.strip().lower()


def normalize_answer(task, text):
    text = text.strip()

    if task == "gsm8k":
        if "####" in text:
            text = text.split("####")[-1].strip()
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return normalize_text(text)

    text = normalize_text(text)

    if task == "rte":
        if "not_entailment" in text:
            return "not_entailment"
        if "entailment" in text:
            return "entailment"
        return text

    if task == "boolq":
        if re.search(r"\b(yes|true)\b", text):
            return "yes"
        if re.search(r"\b(no|false)\b", text):
            return "no"
        return text

    return text


def compute_score(task, prediction, answer):
    normalized_prediction = normalize_answer(task, prediction)
    normalized_answer = normalize_answer(task, answer)
    return int(normalized_prediction == normalized_answer), normalized_prediction, normalized_answer


def summarize(details):
    total = len(details)
    accuracy = sum(item["correct"] for item in details) / total if total > 0 else 0.0
    avg_latency = sum(item["latency"] for item in details) / total if total > 0 else 0.0
    avg_generated_tokens = sum(item["generated_tokens"] for item in details) / total if total > 0 else 0.0

    summary = {
        "num_examples": total,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_generated_tokens": avg_generated_tokens,
    }

    speed_values = [item["tokens_per_second"] for item in details if "tokens_per_second" in item]
    if speed_values:
        summary["avg_tokens_per_second"] = sum(speed_values) / len(speed_values)
        summary["max_tokens_per_second"] = max(speed_values)
        summary["min_tokens_per_second"] = min(speed_values)

    memory_values = [item["peak_memory_mb"] for item in details if "peak_memory_mb" in item]
    if memory_values:
        summary["avg_peak_memory_mb"] = sum(memory_values) / len(memory_values)
        summary["max_peak_memory_mb"] = max(memory_values)

    return summary


def load_eval_data(args):
    data = load_json(args.data_path)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    return data


def save_results(details, summary, save_dir):
    task_name = details[0]["task"] if details else "unknown"
    split_name = details[0]["split"] if details else "unknown"

    save_json(details, os.path.join(save_dir, task_name, f"{split_name}_details.json"))
    save_json(summary, os.path.join(save_dir, task_name, f"{split_name}_summary.json"))


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self.tokenizer = load_tokenizer(args.model_dir)
        self.model = load_model(args.model_dir, self.device)
        self.model.eval()

    def generate_one(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generated_tokens = len(generated_ids)
        latency = end_time - start_time

        result = {
            "prediction": prediction,
            "latency": latency,
            "generated_tokens": generated_tokens,
        }

        if latency > 0:
            result["tokens_per_second"] = generated_tokens / latency

        if self.device == "cuda":
            result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        return result

    def evaluate(self, data):
        details = []

        for example in tqdm(data, desc="Evaluating"):
            prompt = build_prompt(example)
            result = self.generate_one(prompt)
            score, normalized_prediction, normalized_answer = compute_score(
                example["task"], result["prediction"], example["output"]
            )

            row = {
                "task": example["task"],
                "split": example["split"],
                "instruction": example["instruction"],
                "input": example["input"],
                "reference": example["output"],
                "prediction": result["prediction"],
                "normalized_reference": normalized_answer,
                "normalized_prediction": normalized_prediction,
                "correct": score,
                "latency": result["latency"],
                "generated_tokens": result["generated_tokens"],
            }

            if "tokens_per_second" in result:
                row["tokens_per_second"] = result["tokens_per_second"]

            if "peak_memory_mb" in result:
                row["peak_memory_mb"] = result["peak_memory_mb"]

            details.append(row)

        return details

    def run(self):
        data = load_eval_data(self.args)
        details = self.evaluate(data)
        summary = summarize(details)
        save_results(details, summary, self.args.save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
