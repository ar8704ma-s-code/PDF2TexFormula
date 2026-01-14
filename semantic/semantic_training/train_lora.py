import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


# ================== CONFIG ==================

BASE_MODEL = "Qwen/Qwen1.5-0.5B"
TRAIN_DATA_PATH = "/home/baiyinyou/workspace/SmolLatexFormula/src/training/data/train/merged_train.jsonl"
VAL_DATA_PATH = "/home/baiyinyou/workspace/SmolLatexFormula/src/training/data/validation/merged_val.jsonl"
OUTPUT_DIR = "/home/baiyinyou/workspace/SmolLatexFormula/models/Qwen_formula_lora"

MAX_SAMPLES = 2000
MAX_SEQ_LEN = 256


# ================== DATASET ==================

class FormulaDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def build_sample(self, formula: str, correct: bool):
        label = "correct" if correct else "incorrect"

        reason = (
            "The formula is structurally correct."
            if correct else
            "The formula contains one or more issues."
        )

        prompt = f"Verify: {formula}\nCorrectness:"
        target = f" {label}\nReason: {reason}"

        return prompt + target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = self.build_sample(item["formula"], bool(item["correct"]))

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels": enc["input_ids"][0],
        }


def load_jsonl(path, limit=None):
    data = []
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "formula" in obj and "correct" in obj:
                    data.append(obj)
                if limit and len(data) >= limit:
                    break
            except json.JSONDecodeError:
                continue

    print(f"[Loaded] {len(data)} samples from {path}")
    return data


# ================== TRAIN MAIN ==================

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,         # ✔ 修复
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_data = load_jsonl(TRAIN_DATA_PATH, MAX_SAMPLES)
    val_data = load_jsonl(VAL_DATA_PATH, 200)

    train_ds = FormulaDataset(train_data, tokenizer, MAX_SEQ_LEN)
    eval_ds = FormulaDataset(val_data, tokenizer, MAX_SEQ_LEN)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,

        # Training
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3.0,
        learning_rate=1e-4,

        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_torch",

        eval_strategy="steps",
        eval_steps=200,
        logging_steps=25,

        save_steps=200,
        save_total_limit=2,

        bf16=True,

        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[DONE] LoRA model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
