# src/semantic_head.py
# Semantic Head for LaTeX Formula Evaluation
# - Heuristic syntax checks
# - SymPy equivalence checking
# - Explanation-only LLM with LoRA support
# - Semantic verdicts: equivalent / contradiction / non_verifiable

import re
import json
from typing import Optional, Tuple, Dict

import sympy as sp
from sympy.parsing.latex import parse_latex

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # ✅ correct import for LoRA

# =========================================================
# 1. Heuristic checks
# =========================================================

OCR_GARBAGE_PATTERNS = [
    r"\\underline", r"\\rlap", r"\\mit", r"\\bf"
]

INVALID_LATEX_PATTERNS = [
    r"\\backslash\s+[A-Za-z]",
]


def heuristic_check(formula: str) -> Dict:
    for p in OCR_GARBAGE_PATTERNS:
        if re.search(p, formula):
            return {
                "syntactically_correct": False,
                "mathematically_correct": False,
                "semantic_verdict": "garbled",
                "error_type": "garbled_ocr",
                "confidence": 0.95,
                "source": "heuristic",
                "explanation": "The formula contains OCR or encoding artifacts.",
                "suggested_fix": "Clean or retype the LaTeX."
            }

    for p in INVALID_LATEX_PATTERNS:
        if re.search(p, formula):
            return {
                "syntactically_correct": False,
                "mathematically_correct": False,
                "semantic_verdict": "invalid_notation",
                "error_type": "notation_error",
                "confidence": 0.95,
                "source": "heuristic",
                "explanation": "Invalid LaTeX command usage.",
                "suggested_fix": "Replace with valid LaTeX (e.g., \\setminus)."
            }

    return {"pass": True}


# =========================================================
# 2. SymPy helpers
# =========================================================

def normalize_latex_for_sympy(latex: str) -> str:
    s = latex.strip()
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"([a-zA-Z])([a-zA-Z])", r"\1*\2", s)
    s = s.replace("^", "**")
    return s


def split_equation(latex: str) -> Optional[Tuple[str, str]]:
    if latex.count("=") != 1:
        return None
    lhs, rhs = latex.split("=")
    return lhs.strip(), rhs.strip()


def sympy_equivalent(lhs_latex: str, rhs_latex: str) -> Optional[bool]:
    try:
        lhs = parse_latex(lhs_latex)
        rhs = parse_latex(rhs_latex)
        return sp.simplify(lhs - rhs) == 0
    except Exception:
        pass
    try:
        lhs = sp.sympify(normalize_latex_for_sympy(lhs_latex))
        rhs = sp.sympify(normalize_latex_for_sympy(rhs_latex))
        return sp.simplify(lhs - rhs) == 0
    except Exception:
        return None


# =========================================================
# 3. Explanation-only LLM
# =========================================================

class ExplanationLLM:
    def __init__(self, base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
                 lora_dir: Optional[str] = None, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.float16
        )
        if lora_dir:
            model = PeftModel.from_pretrained(model, lora_dir, device_map="auto")
        self.model = model

    def explain(self, formula: str, issue: str) -> str:
        prompt = f"""You are a mathematical LaTeX reviewer.

Formula:
{formula}

Issue:
{issue}

Explain:
- What is happening mathematically
- Why it cannot be verified or is incorrect
- What should be checked or fixed

Do not output True/False.
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # remove prompt echo if present
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return decoded.strip()


# =========================================================
# 4. Formula evaluator
# =========================================================

class FormulaEvaluator:
    def __init__(self, explainer: ExplanationLLM):
        self.explainer = explainer

    def evaluate(self, formula: str) -> Dict:
        fixed_formula = formula  # placeholder for any minor fixes
        # Heuristics
        h = heuristic_check(formula)
        if "pass" not in h:
            return {"original_formula": formula, "fixed_formula": fixed_formula, **h}

        # SymPy check
        eq = split_equation(formula)
        if eq:
            res = sympy_equivalent(eq[0], eq[1])
            if res is True:
                return {
                    "original_formula": formula,
                    "fixed_formula": fixed_formula,
                    "syntactically_correct": True,
                    "mathematically_correct": True,
                    "semantic_verdict": "equivalent",
                    "error_type": None,
                    "confidence": 0.99,
                    "source": "sympy",
                    "explanation": "Both sides are mathematically equivalent.",
                    "suggested_fix": None
                }
            if res is False:
                explanation = self.explainer.explain(
                    formula, "Equality does not hold in general"
                )
                return {
                    "original_formula": formula,
                    "fixed_formula": fixed_formula,
                    "syntactically_correct": True,
                    "mathematically_correct": False,
                    "semantic_verdict": "contradiction",
                    "error_type": "math_error",
                    "confidence": 0.99,
                    "source": "sympy+llm",
                    "explanation": explanation,
                    "suggested_fix": "Correct the mathematical identity."
                }

        # Non-verifiable
        explanation = self.explainer.explain(
            formula, "Expression is not a verifiable equality"
        )
        return {
            "original_formula": formula,
            "fixed_formula": fixed_formula,
            "syntactically_correct": True,
            "mathematically_correct": None,
            "semantic_verdict": "non_verifiable",
            "error_type": "uncertain",
            "confidence": 0.5,
            "source": "llm",
            "explanation": explanation,
            "suggested_fix": "Manual inspection required."
        }


# =========================================================
# 5. Example usage
# =========================================================

if __name__ == "__main__":
    explainer = ExplanationLLM(
        base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        lora_dir="src/repair/repair_training/Repair_improved_1.5B_v1"
    )
    evaluator = FormulaEvaluator(explainer)

    examples = [
        "(a+b)^2 = a^2 + b^2",
        "(a+b)^2 = a^2 + 2ab + b^2",
        "\\Phi : C^0([0,T]) \\to \\mathbb{R}",
        r"r \, \underline{{{\L}}}\, \underline{{{\L}}}\underline{{{\L}}}\rlap/\partial\psi\bigl|\int\bigl(s,\theta\bigr)\,-\,\mathcal{A}_{0}\bigr|\bigl|\infty."
    ]

    for f in examples:
        print("=" * 80)
        print(json.dumps(evaluator.evaluate(f), indent=2, ensure_ascii=False))
