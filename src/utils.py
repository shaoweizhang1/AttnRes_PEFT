import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(example):
    return (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Input:\n"
        f"{example['input']}\n\n"
        "### Response:\n"
    )


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir, device=None):
    if device is None:
        device = get_device()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    if device == "cuda":
        model.to(device)

    return model
