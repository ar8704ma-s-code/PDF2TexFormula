import json
import re
from semantic_head import FormulaEvaluator, ExplanationLLM
from difflib import SequenceMatcher

# -------------------------
# Relaxed string similarity
# -------------------------
def relaxed_match(a: str, b: str, threshold=0.85) -> bool:
    if not a or not b:
        return False
    # normalize whitespace
    a_norm = re.sub(r"\s+", "", a)
    b_norm = re.sub(r"\s+", "", b)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= threshold

# -------------------------
# Benchmark runner
# -------------------------
def run_relaxed_benchmark(json_path: str, output_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["model_1_5B", "model_7B"]
    stats = {m: {"total": 0, "correct": 0} for m in models}

    explainer = ExplanationLLM()  # can provide base_model/lora_dir if needed
    evaluator = FormulaEvaluator(explainer)

    # iterate pages and formulas
    for page, formulas in data["merged_content"].items():
        for entry in formulas:
            gt = entry.get("gt")
            if not gt:
                continue
            for m in models:
                pred = entry.get(m)
                if not pred:
                    continue
                stats[m]["total"] += 1

                # relax: either exact sympy equivalence if equation, or string similarity
                eq = "=" in pred and "=" in gt
                if eq:
                    # attempt sympy equivalence
                    res = evaluator.evaluate(pred)
                    if res.get("semantic_verdict") == "equivalent":
                        stats[m]["correct"] += 1
                    else:
                        # fallback to relaxed string similarity
                        if relaxed_match(pred, gt):
                            stats[m]["correct"] += 1
                else:
                    # for non-equations, use relaxed string similarity
                    if relaxed_match(pred, gt):
                        stats[m]["correct"] += 1

    # compute accuracy
    for m in models:
        total = stats[m]["total"]
        correct = stats[m]["correct"]
        stats[m]["accuracy"] = correct / total if total > 0 else 0.0

    # save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Semantic benchmark results saved to {output_path}")
    print(json.dumps(stats, indent=2))

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to merged_with_gt.json")
    parser.add_argument("--out", default="semantic_benchmark_relaxed.json", help="Output JSON")
    args = parser.parse_args()

    run_relaxed_benchmark(args.json_file, args.out)
