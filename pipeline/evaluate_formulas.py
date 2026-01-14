#evaluate_formulas.py
import re
import json
import os
from typing import List
from semantic_head import FormulaEvaluator, ExplanationLLM  

# -------------------------------------------------
# 0. Preprocessing / Cleaning garbled LaTeX
# -------------------------------------------------
def clean_garbled_latex(formula: str) -> str:
    formula = re.sub(r"\\underline\{\{.*?\}\}", "", formula)  # remove \underline{{...}}
    formula = re.sub(r"\\mit|\\bf", "", formula)               # remove OCR font artifacts
    formula = re.sub(r"\\l\\ell|\\L", "", formula)            # remove garbled letters
    formula = formula.replace(r"\rlap/", "")                  # remove broken \rlap/
    formula = re.sub(r"[\^\_]{2,}", "", formula)             # remove multiple consecutive ^ or _
    formula = re.sub(r"\{\s*\}", "", formula)                # remove empty braces
    formula = formula.strip()
    if len(formula) < 2:
        return ""
    return formula


# -------------------------------------------------
# 1. Extract formulas from LaTeX
# -------------------------------------------------
MATH_ENV_PATTERNS = [
    r"\$(.*?)\$",
    r"\\\((.*?)\\\)",
    r"\\\[(.*?)\\\]",
    r"\\begin\{equation\}(.*?)\\end\{equation\}",
]

def extract_formulas_from_tex(tex_content: str) -> List[str]:
    formulas = []
    for pattern in MATH_ENV_PATTERNS:
        matches = re.findall(pattern, tex_content, re.DOTALL)
        formulas.extend([m.strip() for m in matches])
    return formulas

# -------------------------------------------------
# 2. Automatic minor LaTeX fixes
# -------------------------------------------------
def fix_latex(formula: str) -> str:
    # Common minor fixes
    formula = re.sub(r"\\backslash\s*([A-Za-z])", r"\\setminus \1", formula)
    formula = formula.replace(r"\implies", r"\Rightarrow")
    return formula

# -------------------------------------------------
# 3. Evaluate formulas and store results
# -------------------------------------------------
def evaluate_tex_file(tex_path: str, output_dir: str, base_model=None, lora_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    formulas = extract_formulas_from_tex(tex_content)
    print(f"Found {len(formulas)} formulas.")

    explainer = ExplanationLLM(base_model=base_model, lora_dir=lora_dir)
    evaluator = FormulaEvaluator(explainer)

    results = []
    for formula in formulas:
        cleaned = clean_garbled_latex(formula)
        if not cleaned:
            print(f"Skipping garbled formula: {formula[:30]}...")
            continue
        fixed_formula = fix_latex(cleaned)
        res = evaluator.evaluate(fixed_formula)
        res_entry = {
            "original_formula": formula,
            "fixed_formula": fixed_formula,
            **res
        }
        results.append(res_entry)

    # Save as JSON
    json_path = os.path.join(output_dir, "results_final.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save as TXT summary
    txt_path = os.path.join(output_dir, "results_final.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Original Formula: {r['original_formula']}\n")
            f.write(f"Fixed Formula: {r['fixed_formula']}\n")
            f.write(f"Syntactically correct: {r['syntactically_correct']}\n")
            f.write(f"Mathematically correct: {r['mathematically_correct']}\n")
            f.write(f"Error type: {r['error_type']}\n")
            f.write(f"Confidence: {r['confidence']}\n")
            f.write(f"Source: {r['source']}\n")
            f.write(f"Suggested fix: {r.get('suggested_fix')}\n")
            f.write(f"Explanation:\n{r['explanation']}\n")
            f.write("="*80 + "\n")

    print(f"Results saved to {json_path} and {txt_path}")

# -------------------------------------------------
# 4. CLI usage
# -------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate formulas in a LaTeX file with automatic minor fixes and optional LoRA model."
    )
    parser.add_argument("tex_file", help="Path to the LaTeX file")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model name")
    parser.add_argument("--lora_dir", default=None,
                        help="LoRA directory for fine-tuned model")

    args = parser.parse_args()

    evaluate_tex_file(
        tex_path=args.tex_file,
        output_dir=args.out,
        base_model=args.base_model,
        lora_dir=args.lora_dir
    )