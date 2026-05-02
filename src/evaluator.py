import argparse
import os
import re
import time

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig

from src.AttnResAdapter import load_qwen3_attnres_model
from src.utils import get_device, build_prompt, load_json, load_model, load_tokenizer, save_json


DEFAULT_MODEL_DIR = "./model"
DEFAULT_SAVE_DIR = "./results"


def normalize_text(text):
    return text.strip().lower()


def _label_from_text(text, labels, aliases=None):
    """
    Find the first label mentioned in text using word-boundary-safe matching.
    Prevents matching labels that appear inside longer underscore-joined tokens
    (e.g. "yes_or_no" should not match "yes", "_not_entailment" should not match "not_entailment").
    """
    text = normalize_text(text).replace("-", "_").replace(" ", "_")
    aliases = aliases or {}
    for label, terms in aliases.items():
        for term in terms:
            if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text):
                return label
    for label in sorted(labels, key=len, reverse=True):
        normalized = label.lower().replace("-", "_").replace(" ", "_")
        if re.search(rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])", text):
            return label
    return text


def normalize_answer(task, text):
    text = text.strip()

    if task == "gsm8k":
        if "####" in text:
            text = text.split("####")[-1].strip()
        numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if numbers:
            return numbers[-1].replace(",", "")
        return normalize_text(text)

    if task == "boolq":
        return _label_from_text(
            text,
            ["no", "yes"],
            aliases={
                "yes": ["yes", "true", "correct"],
                "no":  ["no", "false", "incorrect"],
            },
        )

    if task == "rte":
        return _label_from_text(
            text,
            ["not_entailment", "entailment"],
            aliases={
                "not_entailment": ["not_entailment", "not_entailed", "not", "false"],
                "entailment":     ["entailment", "entailed", "yes", "true"],
            },
        )

    return normalize_text(text)


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
        # Lazy import so missing vllm doesn't crash transformers-only usage
        from vllm import LLM, SamplingParams

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
        torch_dtype = torch.float32 if self.args.model_dtype == "fp32" else torch.float16

        if self.args.method == "base":
            self.model = load_model(self.args.model_dir, self.device, torch_dtype=torch_dtype)
        elif self.args.method == "lora":
            if self.args.adapter_dir is None:
                raise ValueError("LoRA evaluation requires --adapter_dir.")
            base_model = load_model(self.args.base_model_dir, self.device, torch_dtype=torch_dtype)
            self.model = PeftModel.from_pretrained(base_model, self.args.adapter_dir)
        elif self.args.method == "attnres":
            if self.args.adapter_dir is None:
                raise ValueError("AttnRes evaluation requires --adapter_dir.")
            base_model = load_model(self.args.base_model_dir, self.device, torch_dtype=torch_dtype)
            self.model = load_qwen3_attnres_model(
                base_model,
                lookback=self.args.attnres_lookback,
            )
            state_dict = load_attnres_state_dict(self.args.adapter_dir, self.device)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            adapter_missing = [k for k in missing if k.startswith("adapters.")]
            if adapter_missing:
                print(f"[WARNING] {len(adapter_missing)} adapter keys missing from checkpoint: {adapter_missing[:5]}")
            if unexpected:
                print(f"[WARNING] {len(unexpected)} unexpected keys in checkpoint: {unexpected[:5]}")
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
            result = {"prediction": prediction, "latency": latency, "generated_tokens": generated_tokens}
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

        total_latency = end_time - start_time
        per_sample_latency = total_latency / len(outputs) if len(outputs) > 0 else 0.0
        # With left padding, HF returns [padded_input | generated] for every sequence.
        # Slice by padded input width (uniform across the batch), NOT by per-sample
        # real token count — otherwise prompt tail leaks into the prediction string.
        input_width = inputs["input_ids"].shape[1]

        results = []
        for output_ids in outputs:
            generated_ids = output_ids[input_width:]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generated_tokens = len(generated_ids)

            result = {
                "prediction": prediction,
                "latency": per_sample_latency,
                "generated_tokens": generated_tokens,
            }
            if per_sample_latency > 0:
                result["tokens_per_second"] = generated_tokens / per_sample_latency
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
                score, norm_pred, norm_ans = compute_score(
                    example["task"], result["prediction"], example["output"]
                )
                row = {
                    "task": example["task"],
                    "split": example["split"],
                    "instruction": example["instruction"],
                    "input": example["input"],
                    "reference": example["output"],
                    "prediction": result["prediction"],
                    "normalized_reference": norm_ans,
                    "normalized_prediction": norm_pred,
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
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--model_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--attnres_lookback", type=int, default=8)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
