import argparse
import os
import re
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import LLM, SamplingParams

from src.AttnResAdapter import load_qwen3_attnres_model
from src.utils import get_device, build_prompt, load_json, load_model, load_tokenizer, save_json


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


def load_attnres_state_dict(adapter_dir, device):
    safetensors_path = os.path.join(adapter_dir, "model.safetensors")
    pytorch_path = os.path.join(adapter_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device=device)

    if os.path.exists(pytorch_path):
        return torch.load(pytorch_path, map_location=device)

    raise ValueError(f"Cannot find AttnRes checkpoint in {adapter_dir}")


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = load_tokenizer(args.model_dir if args.method == "base" else args.base_model_dir)

        if self.args.backend == "vllm":
            self.setup_vllm()
        else:
            self.setup_transformers()

    def setup_vllm(self):
        if self.args.method != "base":
            raise ValueError("vllm backend currently only supports --method base.")

        self.model = LLM(
            model=self.args.model_dir,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.args.max_new_tokens,
        )

    def setup_transformers(self):
        self.device = get_device()

        if self.args.method == "base":
            self.model = load_model(self.args.model_dir, self.device)
        elif self.args.method == "lora":
            if self.args.adapter_dir is None:
                raise ValueError("LoRA evaluation requires --adapter_dir.")
            base_model = load_model(self.args.base_model_dir, self.device)
            self.model = PeftModel.from_pretrained(base_model, self.args.adapter_dir)
        elif self.args.method == "attnres":
            if self.args.adapter_dir is None:
                raise ValueError("AttnRes evaluation requires --adapter_dir.")
            base_model = load_model(self.args.base_model_dir, self.device)
            self.model = load_qwen3_attnres_model(
                base_model,
                lookback=self.args.attnres_lookback,
                gate_init=self.args.attnres_gate_init,
            )
            state_dict = load_attnres_state_dict(self.args.adapter_dir, self.device)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Unsupported method: {self.args.method}")

        self.model.eval()
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generate_batch_vllm(self, prompts):
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

    def generate_batch_transformers(self, prompts):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
        ).to(self.device)

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
            )
        end_time = time.time()

        results = []
        latency = end_time - start_time
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        for i, output_ids in enumerate(outputs):
            generated_ids = output_ids[int(prompt_lengths[i]):]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_tokens = len(generated_ids)

            result = {
                "prediction": prediction,
                "latency": latency,
                "generated_tokens": generated_tokens,
            }

            if latency > 0:
                result["tokens_per_second"] = generated_tokens / latency

            if self.device == "cuda":
                result["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

            results.append(result)

        return results

    def generate_batch(self, prompts):
        if self.args.backend == "vllm":
            return self.generate_batch_vllm(prompts)
        return self.generate_batch_transformers(prompts)

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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="transformers")
    parser.add_argument("--method", choices=["base", "lora", "attnres"], default="base")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base_model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--attnres_lookback", type=int, default=8)
    parser.add_argument("--attnres_gate_init", type=float, default=0.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
