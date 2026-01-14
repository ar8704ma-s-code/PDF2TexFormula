# generate_pseudo_labels.py
import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIGURATION
# ============================================================
ALIGNED_IN_DIR = Path("data/aligned_dataset_filtered")
ALIGNED_OUT_DIR = Path("data/aligned_dataset_refined")
ALIGNED_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 12  # Adjust based on your GPU memory



# ============================================================
# LLM REPAIR HEAD
# ============================================================
class RepairHead:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[RepairHead] Loading model on {device}...")

        # Critical fix for decoder-only models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",  # Essential for proper generation
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[RepairHead] Ready.\n")

    def build_prompt(self, expr: str) -> str:
        """Build structured prompt for LaTeX repair task"""
        return (
            "<|im_start|>system\n"
            "You are a LaTeX repair classifier. You must output EXACTLY one of two options:\n"
            "1. If the input is meaningful mathematics (equations, formulas, expressions), output the repaired LaTeX\n"
            "2. If the input is noise (labels, numbers, references, section markers, isolated symbols), output EXACTLY: noise\n"
            "\n"
            "Examples of noise:\n"
            "- \"(1)\", \"(2.3)\", \"[A.1]\", \"\\textbf{(1)}\", \"\\mathbf{(1)}\", \"1\", \"A.1\"\n"
            "- \"\\ref{eq1}\", \"\\label{sec2}\", \"\\cite{author2023}\"\n"
            "- \"+\", \"=\", \"*\" (isolated symbols)\n"
            "\n"
            "Rules:\n"
            "- NEVER output anything else\n"
            "- NEVER add explanations\n"
            "- For noise, output EXACTLY: noise\n"
            "- For math, output ONLY the repaired LaTeX\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{expr}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def repair_batch(self, expr_list: List[str]) -> List[str]:
        """Batch process expressions through LLM for repair"""
        prompts = [self.build_prompt(x) for x in expr_list]
        
        # Tokenize inputs with left padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)

        # Generate repairs with constrained output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # Short outputs: either "noise" or brief LaTeX
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                temperature=0.1,  # More deterministic outputs
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

        # Extract assistant responses strictly
        answers = []
        for text in decoded:
            # Find assistant section in ChatML format
            m = re.findall(r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)", text, flags=re.S)
            if m:
                ans = m[0].strip()
                # Check if LLM explicitly returned "noise"
                if re.match(r'^noise\b', ans, re.IGNORECASE):
                    answers.append("noise")
                else:
                    # Otherwise assume it's repaired LaTeX
                    answers.append(ans)
            else:
                # Fallback: check entire output
                if "noise" in text.lower():
                    answers.append("noise")
                else:
                    answers.append(text.strip())
        
        return answers


# ============================================================
# NOISE DETECTION
# ============================================================
def is_trivial(expr: str) -> bool:
    """Determine if expression is trivial noise (not meaningful math)"""
    if not expr:
        return True
    
    # Case 1: LLM explicitly returned "noise"
    if expr.lower().strip() == "noise":
        return True
    
    # Case 2: Common noise patterns in LaTeX
    expr_clean = expr.strip()
    
    latex_noise_patterns = [
        r'^\\mathbf\{\s*\(?\s*\d+\s*\)?\s*\}$',  # \mathbf{(1)}, \mathbf{1}
        r'^\\textbf\{\s*\(?\s*\d+\s*\)?\s*\}$',  # \textbf{(1)}, \textbf{1}
        r'^\\mathit\{\s*\(?\s*\d+\s*\)?\s*\}$',  # \mathit{(1)}
        r'^\\mathrm\{\s*\(?\s*\d+\s*\)?\s*\}$',  # \mathrm{(1)}
        r'^\\label\{.*\}$',                      # \label{anything}
        r'^\\ref\{.*\}$',                        # \ref{anything}
        r'^\\eqref\{.*\}$',                      # \eqref{anything}
        r'^\\cite\{.*\}$',                       # \cite{anything}
        r'^\(?\d+(\.\d+)*\)?$',                  # (1), 2, (3.4)
        r'^\[?\d+(\.\d+)*\]?$',                  # [1], [2.3]
        r'^[A-Z]\.\d+$',                         # A.1, B.2
        r'^[+\-*/=]$',                           # Single operators
    ]
    
    for pattern in latex_noise_patterns:
        if re.match(pattern, expr_clean):
            return True
    
    # Case 3: Very short expressions without mathematical content
    if len(expr_clean) < 6 and not re.search(r'[a-zA-Z]{2,}', expr_clean):
        return True
    
    return False


# ============================================================
# FILE PROCESSING
# ============================================================
def process_file(path: Path, repair_head: RepairHead):
    """Process single file: convert repair_no_gt to repair_pseudo_gt or noise"""
    out_path = ALIGNED_OUT_DIR / path.name

    if out_path.exists():
        print(f"[Skip] {path.name}")
        return

    # Read input file
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    # Prepare output file
    fout = out_path.open("w", encoding="utf-8")

    # Collect expressions needing repair
    exprs_to_repair = []
    indices_to_repair = []

    # First pass: separate repair_no_gt from other tasks
    for idx, record in enumerate(lines):
        if record["task"] == "repair_no_gt":
            exprs_to_repair.append(record["raw_ocr"])
            indices_to_repair.append(idx)
        else:
            # Write with_gt and noise records directly (unchanged)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  → Repairing {len(exprs_to_repair)} raw expressions...")

    repaired_count = 0
    noise_count = 0

    # Process in batches
    for start_idx in range(0, len(exprs_to_repair), BATCH_SIZE):
        batch_expr = exprs_to_repair[start_idx:start_idx + BATCH_SIZE]
        batch_idx = indices_to_repair[start_idx:start_idx + BATCH_SIZE]

        # Get LLM repairs
        outputs = repair_head.repair_batch(batch_expr)

        # Update records based on LLM output
        for record_idx, output in zip(batch_idx, outputs):
            record = lines[record_idx]

            if is_trivial(output):
                # Convert to noise
                record["task"] = "noise"
                record["gt_latex"] = None
                noise_count += 1
            else:
                # Convert to repair_pseudo_gt with generated ground truth
                record["task"] = "repair_pseudo_gt"
                record["gt_latex"] = output
                repaired_count += 1

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    fout.close()
    print(f"[Done] {path.name} - {repaired_count} repaired, {noise_count} noise")


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    # Set GPU device
    torch.cuda.set_device(0)
    
    # Initialize repair head
    repair_head = RepairHead(MODEL_NAME)

    # Get all files to process
    files = sorted(ALIGNED_IN_DIR.glob("*.jsonl"))
    print(f"Found {len(files)} files.")

    # Process each file
    for file_path in tqdm(files, desc="Generating pseudo labels"):
        process_file(file_path, repair_head)

    print("\nAll files processed.")


if __name__ == "__main__":
    import os
    # Set memory optimization for PyTorch
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()