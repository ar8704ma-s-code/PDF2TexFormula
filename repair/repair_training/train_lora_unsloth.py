#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA fine-tuning script for LaTeX repair + noise detection
Model: Qwen/Qwen2.5-Math-7B-Instruct
Backend: Unsloth FastLanguageModel + HF Trainer
Dataset format: {"input": "...", "output": "..."} in JSONL
"""

# ------------------------------------------------------
# MUST IMPORT UNSLOTH FIRST (required for patching)
# ------------------------------------------------------
import os

# Disable PyTorch Inductor (for clusters without nvcc permissions)
os.environ["TORCHINDUCTOR_DISABLED"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from unsloth import FastLanguageModel
import json
from typing import List, Dict

import torch
from datasets import Dataset
from packaging import version
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------
# Transformers version check
# ------------------------------------------------------
MIN_VERSION = "4.47.0"
if version.parse(transformers.__version__) < version.parse(MIN_VERSION):
    raise RuntimeError(
        f"transformers>={MIN_VERSION} required, found {transformers.__version__}"
    )

# ------------------------------------------------------
# Paths
# ------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"

WITH_GT_PATH = "data/train_dataset_cleaned/with_gt.jsonl"
SHORT_NOISE_PATH = "data/train_dataset_cleaned/short_noise.jsonl"
LONG_NOISE_PATH = "data/train_dataset_cleaned/long_noise.jsonl"

OUTPUT_DIR = "repair_lora_short_long_v4"
MAX_LENGTH = 512

# ------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------
SYSTEM_PROMPT = """You are a LaTeX formula repair specialist.

Your tasks:

1. If the input is MEANINGFUL mathematics:
   - Repair OCR errors
   - Fix LaTeX syntax and bracket structure
   - Preserve the mathematical meaning
   - Output ONLY the repaired LaTeX

2. If the input is NOISE:
   - Output exactly: noise

Strict rules:
- Do not invent new math
- Do not add explanations
- Output ONLY repaired LaTeX or "noise".
"""

# ------------------------------------------------------
# Build ChatML prompt
# ------------------------------------------------------
def build_chatml_prompt(raw_text, target):
    return (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT
        + "<|im_end|>\n"
        "<|im_start|>user\nInput (OCR): "
        + raw_text
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        + target
        + "<|im_end|>\n"
    )

# ------------------------------------------------------
# Load JSONL
# ------------------------------------------------------
def load_records(jsonl_path, noise_weight=1.0, math_weight=1.0):
    records = []
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            raw = obj["input"]
            out = obj["output"]
            is_noise = isinstance(out, str) and out.strip().lower() == "noise"
            w = noise_weight if is_noise else math_weight
            prompt = build_chatml_prompt(raw, out)
            records.append({"text": prompt, "weight": float(w)})
    return records

def load_all_records():
    rec = []
    rec += load_records(WITH_GT_PATH, noise_weight=1.0, math_weight=1.0)
    rec += load_records(SHORT_NOISE_PATH, noise_weight=1.0)
    rec += load_records(LONG_NOISE_PATH, noise_weight=0.3)
    return rec

# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
print("Loading dataset...")
records = load_all_records()
dataset = Dataset.from_list(records)
dataset = dataset.filter(lambda x: len(x["text"]) > 10)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = dataset["train"], dataset["test"]

print("Train:", len(train_ds), "Eval:", len(eval_ds))

# ------------------------------------------------------
# Load model
# ------------------------------------------------------
print("Loading model via Unsloth...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    dtype=None,
    load_in_4bit=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------
# LoRA config
# ------------------------------------------------------
lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_cfg.r,
    lora_alpha=lora_cfg.lora_alpha,
    lora_dropout=lora_cfg.lora_dropout,
    target_modules=lora_cfg.target_modules,
    bias=lora_cfg.bias,
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# ------------------------------------------------------
# Tokenization
# ------------------------------------------------------
def tokenize_fn(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    enc["weight"] = batch["weight"]
    return enc

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_ds = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# ------------------------------------------------------
# Weighted Trainer (Unsloth-compatible)
# ------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # <-- REQUIRED for Unsloth
        **kwargs,
    ):
        labels = inputs["input_ids"].clone()
        w = inputs.get("weight", None)

        if w is not None:
            scalar_w = w.mean()
        else:
            scalar_w = torch.tensor(1.0, device=labels.device)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels,
        )

        loss = outputs.loss * scalar_w
        return (loss, outputs) if return_outputs else loss

# ------------------------------------------------------
# TrainingArguments
# ------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,

    bf16=torch.cuda.is_available(),

    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,

    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ------------------------------------------------------
# Train
# ------------------------------------------------------
def main():
    print("Starting training...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
