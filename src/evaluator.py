import argparse
import os
import re
import time

from tqdm import tqdm
from src.utils import build_prompt, load_json, load_tokenizer, save_json
from vllm import LLM, SamplingParams


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
        self.tokenizer = load_tokenizer(args.model_dir)
        self.model = LLM(
            model=args.model_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_new_tokens,
        )

    def generate_batch(self, prompts):
        start_time = time.time()
        outputs = self.model.generate(prompts, self.sampling_params)
        end_time = time.time()

        results = []
        latency = end_time - start_time

        for output in outputs:
            prediction = output.outputs[0].text.strip()
            generated_tokens = len(output.outputs[0].token_ids)

            result = {
                "prediction": prediction,
                "latency": latency,
                "generated_tokens": generated_tokens,
            }

            if latency > 0:
                result["tokens_per_second"] = generated_tokens / latency

            results.append(result)

        return results

    def evaluate(self, data):
        details = []

        for i in tqdm(range(0, len(data), self.args.batch_size), desc="Evaluating"):
            batch = data[i:i + self.args.batch_size]
            prompts = [build_prompt(example) for example in batch]
            results = self.generate_batch(prompts)

            for example, result in zip(batch, results):
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
