#!/usr/bin/env python3
"""
Formula evaluation — LLM-assisted correction version:
- Detects trivial numbers or labels
- Detects malformed LaTeX
- Automatically asks LLM to correct syntax errors
- Saves results in JSONL and TXT formats
"""

import os
import argparse
import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re

# ==============================================================
# Utility: Detect malformed LaTeX
# ==============================================================
def is_malformed_latex(formula):
    """Return True if braces are unbalanced or brackets mismatched."""
    stack = []
    for c in formula:
        if c in "{[(":
            stack.append(c)
        elif c in "}])":
            if not stack:
                return True
            last = stack.pop()
            if (last, c) not in {("{", "}"), ("[", "]"), ("(", ")")}:
                return True
    return len(stack) != 0

# ==============================================================
# Utility: Detect trivial numbers or labels
# ==============================================================
def looks_like_number(formula):
    """Detect integers, floats, or trivial numbers inside math."""
    stripped = formula.strip()
    if re.fullmatch(r"\d+", stripped):
        return True
    if re.fullmatch(r"\(?\d+(\.\d+)?\)?", stripped):
        return True
    if re.fullmatch(r"(\\angle\s*)?\d+", stripped):
        return True
    return False

def has_math_semantics(formula):
    """Check if expression has math structure beyond trivial numbers."""
    math_indicators = [
        r"\\", r"\+", r"-", r"=", r"\^",
        r"\\frac", r"\\vec", r"\\sigma",
        r"\{.*\}", r"\[.*\]",
        r"[a-zA-Z]{2,}",
    ]
    return any(re.search(p, formula) for p in math_indicators)

# ==============================================================
# Formula Evaluator
# ==============================================================
class FormulaEvaluator:
    def __init__(self, model_path, base_model="Qwen/Qwen1.5-0.5B"):
        print("[Loading tokenizer]")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[Loading base model]")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("[Loading LoRA adapter]")
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

    def build_prompt(self, formula):
        return (
            "You are an expert in mathematical LaTeX formula verification.\n"
            "Determine if the following formula is correct.\n\n"
            f"Formula:\n{formula}\n\n"
            "Answer in exactly this format:\n"
            "Correct: yes/no\n"
            "Reason: <short explanation>\n"
            "Correction: <corrected formula if incorrect, otherwise repeat original>\n"
        )

    def evaluate(self, formula):
        # 1. Malformed syntax → let LLM correct it
        if is_malformed_latex(formula):
            prompt = (
                "You are an expert in LaTeX math.\n"
                "The following formula has syntax errors (unbalanced or mismatched brackets/parentheses).\n"
                "Provide a corrected LaTeX formula and explain the correction briefly.\n\n"
                f"Formula:\n{formula}\n\n"
                "Answer in exactly this format:\n"
                "Corrected: <corrected formula>\n"
                "Reason: <short explanation of the correction>\n"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(out[0], skip_special_tokens=True)
            corr_match = re.search(r"Corrected:\s*(.*)", response, re.I)
            corrected_formula = corr_match.group(1).strip() if corr_match else None
            reason_match = re.search(r"Reason:\s*(.*)", response, re.I)
            reason = reason_match.group(1).strip() if reason_match else "No explanation provided."

            return {
                "formula": formula,
                "is_correct": False,
                "error_type": "syntax_error",
                "reason": reason,
                "suggested_correction": corrected_formula
            }

        # 2. Trivial numbers or labels → semantic error
        if looks_like_number(formula) or not has_math_semantics(formula):
            return {
                "formula": formula,
                "is_correct": False,
                "error_type": "semantic_error",
                "reason": "This is not a formula, only a trivial number/label or lacks math structure.",
                "suggested_correction": None
            }

        # 3. Evaluate formula normally using the model
        prompt = self.build_prompt(formula)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self.parse_model_response(response, formula)

    def parse_model_response(self, response, formula):
        text = response.lower()
        is_correct = "correct: yes" in text
        reason_match = re.search(r"reason:(.*)", response, re.I | re.S)
        reason = reason_match.group(1).strip() if reason_match else "No explanation provided."
        corr_match = re.search(r"correction:(.*)", response, re.I | re.S)
        correction = corr_match.group(1).strip() if corr_match else None
        if is_correct:
            correction = None
        return {
            "formula": formula,
            "is_correct": is_correct,
            "error_type": "none" if is_correct else "model_detected_error",
            "reason": reason,
            "suggested_correction": correction
        }

# ==============================================================
# Formula Extraction
# ==============================================================
def load_formulas(input_path):
    text = Path(input_path).read_text(encoding="utf-8")
    formulas = []
    formulas += re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)
    formulas += re.findall(r"\\\((.*?)\\\)", text, flags=re.S)
    formulas += re.findall(r"\\\[(.*?)\\\]", text, flags=re.S)
    formulas += re.findall(r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", text, flags=re.S)
    for f in re.findall(r"\$(.*?)\$", text, flags=re.S):
        if len(f.strip()) > 1:
            formulas.append(f)
    clean = [f.strip() for f in formulas if f.strip()]
    clean = list(dict.fromkeys(clean))
    print(f"[Extractor] Found {len(clean)} unique formulas.")
    return clean

# ==============================================================
# Save results
# ==============================================================
def save_results(results, jsonl_path):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    txt_path = jsonl_path.replace(".jsonl", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Formula: {r['formula']}\n")
            f.write(f"Correct: {r['is_correct']}\n")
            f.write(f"Error Type: {r['error_type']}\n")
            if r["suggested_correction"]:
                f.write(f"Suggested Correction: {r['suggested_correction']}\n")
            f.write(f"Reason: {r['reason']}\n")
            f.write("-" * 50 + "\n")
    print(f"[Saved] JSONL → {jsonl_path}")
    print(f"[Saved] TXT   → {txt_path}")

# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen1.5-0.5B")
    args = parser.parse_args()

    evaluator = FormulaEvaluator(args.model_path, args.base_model)
    formulas = load_formulas(args.input)
    results = [evaluator.evaluate(f) for f in tqdm(formulas)]
    save_results(results, args.output)

if __name__ == "__main__":
    main()
