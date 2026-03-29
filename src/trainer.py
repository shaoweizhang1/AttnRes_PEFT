import argparse
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

from src.logger import Logger
from src.utils import build_prompt, load_json, load_model, load_tokenizer, save_json


DEFAULT_MODEL_DIR = "./model"
DEFAULT_SAVE_DIR = "./checkpoints"


def preprocess_data(data, tokenizer, max_length):
    processed_data = []

    for example in tqdm(data, desc="Tokenizing"):
        prompt = build_prompt(example)
        full_text = prompt + example["output"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]

        labels = full_ids[:]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels[:prompt_length] = [-100] * prompt_length

        processed_data.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )

    return Dataset.from_list(processed_data)


def collate_fn(batch):
    pad_token_id = 0
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def get_lora_model(model, args):
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# def get_attnres_model(model, args):
#     from src.attnres_peft.tuner import get_attnres_model

#     model = get_attnres_model(model, args)
#     if hasattr(model, "print_trainable_parameters"):
#         model.print_trainable_parameters()
#     return model

def get_attnres_model(model, args):
    from src.AttnResAdapter import load_qwen3_attnres_model

    model = load_qwen3_attnres_model(
        model,
        lookback=args.attnres_lookback,
        gate_init=args.attnres_gate_init,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


class TrainerRunner:
    def __init__(self, args):
        self.args = args
        self.logger = Logger(args)

    def load_model_and_tokenizer(self):
        tokenizer = load_tokenizer(self.args.model_dir)
        model = load_model(self.args.model_dir)

        if self.args.method == "lora":
            model = get_lora_model(model, self.args)
        else:
            model = get_attnres_model(model, self.args)

        return model, tokenizer

    def load_datasets(self, tokenizer):
        train_data = load_json(self.args.train_path)
        train_dataset = preprocess_data(train_data, tokenizer, self.args.max_length)

        eval_dataset = None
        if self.args.val_path:
            val_data = load_json(self.args.val_path)
            eval_dataset = preprocess_data(val_data, tokenizer, self.args.max_length)

        return train_dataset, eval_dataset

    def build_training_args(self, has_eval):
        return TrainingArguments(
            output_dir=self.args.save_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_steps=self.args.eval_steps,
            eval_strategy="steps" if has_eval else "no",
            save_strategy="steps",
            logging_strategy="steps",
            fp16=torch.cuda.is_available(),
            report_to="wandb" if self.args.use_wandb else "none",
            run_name=self.args.wandb_run_name,
            remove_unused_columns=False,
        )

    def run(self):
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_json(vars(self.args), os.path.join(self.args.save_dir, "train_args.json"))
        self.logger.setup_wandb()

        model, tokenizer = self.load_model_and_tokenizer()
        train_dataset, eval_dataset = self.load_datasets(tokenizer)
        training_args = self.build_training_args(eval_dataset is not None)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collate_fn,
        )

        trainer.train()
        trainer.save_model(self.args.save_dir)
        tokenizer.save_pretrained(self.args.save_dir)
        self.logger.finish()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "attnres"], required=True)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", default=None)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--attnres_lookback", type=int, default=8)
    parser.add_argument("--attnres_gate_init", type=float, default=0.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    trainer_runner = TrainerRunner(args)
    trainer_runner.run()


if __name__ == "__main__":
    main()
