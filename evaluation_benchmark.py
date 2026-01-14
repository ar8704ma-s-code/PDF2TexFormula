import json
from collections import Counter, defaultdict
from typing import Dict, Tuple, Any

# ============================================================
# Semantic explanation buckets (tightened)
# ============================================================
def bucket_explanation(expl: str, verdict: str) -> str:
    if not expl:
        return "incoherent"

    expl_l = expl.lower()

    if len(expl_l) < 40:
        return "incoherent"

    if any(k in expl_l for k in [
        "manual inspection",
        "cannot be verified",
        "insufficient information",
        "not enough information"
    ]):
        return "underdefined"

    if verdict == "non_verifiable" and any(k in expl_l for k in [
        "mathematically correct",
        "equivalent",
        "holds"
    ]):
        return "overclaim"

    if any(k in expl_l for k in [
        "equivalent",
        "mathematically correct",
        "holds"
    ]):
        return "equivalent"

    if any(k in expl_l for k in [
        "contradiction",
        "incorrect",
        "does not hold",
        "false"
    ]):
        return "contradiction"

    return "other"


# ============================================================
# Ground-truth semantic label
# ============================================================
def gt_semantic_label(gt_formula: str) -> str:
    if not gt_formula:
        return "unknown"

    gt = gt_formula.lower()

    if "=" in gt or "\\le" in gt or "\\ge" in gt:
        return "assertion"

    return "definition"


# ============================================================
# Strict vs relaxed matching
# ============================================================
STRICT_MATCH = {
    ("equivalent", "assertion"),
    ("contradiction", "assertion"),
    ("underdefined", "definition"),
}

RELAXED_MATCH = {
    ("underdefined", "assertion"),
    ("equivalent", "definition"),
}


def strict_match(bucket: str, gt_label: str) -> bool:
    return (bucket, gt_label) in STRICT_MATCH


def relaxed_match(bucket: str, gt_label: str) -> bool:
    return strict_match(bucket, gt_label) or (bucket, gt_label) in RELAXED_MATCH


# ============================================================
# Robust GT loader
# ============================================================
def load_gt_map(gt_json_path: str) -> Dict[Tuple[str, str], str]:
    """
    Returns:
        {(page, formula): gt_formula}
    Handles list/dict/garbage safely.
    """
    gt_data = json.load(open(gt_json_path, "r", encoding="utf-8"))
    gt_map = {}

    def process_entry(entry: Any):
        if not isinstance(entry, dict):
            return

        page = entry.get("page")
        formula = entry.get("formula")
        gt_formula = (
            entry.get("gt_formula")
            or entry.get("original_formula")
            or entry.get("fixed_formula")
        )

        if page and formula:
            gt_map[(page, formula)] = gt_formula

    # Case 1: list of dicts
    if isinstance(gt_data, list):
        for entry in gt_data:
            process_entry(entry)

    # Case 2: dict of entries
    elif isinstance(gt_data, dict):
        for value in gt_data.values():
            if isinstance(value, list):
                for entry in value:
                    process_entry(entry)
            else:
                process_entry(value)

    return gt_map


# ============================================================
# Benchmark runner
# ============================================================
def run_benchmark(eval_json, gt_json, out_json, out_latex):
    eval_data = json.load(open(eval_json, "r", encoding="utf-8"))
    gt_map = load_gt_map(gt_json)

    bucket_stats = defaultdict(lambda: {
        "count": 0,
        "strict_correct": 0,
        "relaxed_correct": 0
    })

    total = 0
    strict_correct_total = 0
    relaxed_correct_total = 0

    for entry in eval_data:
        if not isinstance(entry, dict):
            continue

        expl = entry.get("explanation", "")
        verdict = entry.get("semantic_verdict", "")
        page = entry.get("page")
        formula = entry.get("formula")

        if not expl or not verdict or not page or not formula:
            continue

        bucket = bucket_explanation(expl, verdict)
        bucket_stats[bucket]["count"] += 1

        gt_formula = gt_map.get((page, formula))
        gt_label = gt_semantic_label(gt_formula)

        if strict_match(bucket, gt_label):
            strict_correct_total += 1
            bucket_stats[bucket]["strict_correct"] += 1

        if relaxed_match(bucket, gt_label):
            relaxed_correct_total += 1
            bucket_stats[bucket]["relaxed_correct"] += 1

        total += 1

    # ========================================================
    # Results
    # ========================================================
    results = {
        "total": total,
        "strict_accuracy_vs_gt": round(100 * strict_correct_total / total, 2) if total else 0.0,
        "relaxed_accuracy_vs_gt": round(100 * relaxed_correct_total / total, 2) if total else 0.0,
        "buckets": {}
    }

    for bucket, stats in bucket_stats.items():
        count = stats["count"]
        results["buckets"][bucket] = {
            "count": count,
            "percentage": round(100 * count / total, 2) if total else 0.0,
            "strict_accuracy": round(100 * stats["strict_correct"] / count, 2) if count else 0.0,
            "relaxed_accuracy": round(100 * stats["relaxed_correct"] / count, 2) if count else 0.0,
        }

    json.dump(results, open(out_json, "w"), indent=2)

    # ========================================================
    # LaTeX table
    # ========================================================
    with open(out_latex, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\hline
Semantic Bucket & Count & \% & Strict Acc (\%) & Relaxed Acc (\%) \\
\hline
""")

        for bucket, stats in results["buckets"].items():
            f.write(
                f"{bucket.replace('_',' ')} & "
                f"{stats['count']} & "
                f"{stats['percentage']} & "
                f"{stats['strict_accuracy']} & "
                f"{stats['relaxed_accuracy']} \\\\\n"
            )

        f.write(r"""\hline
\end{tabular}
\caption{Semantic explanation quality vs ground-truth semantics.
Strict accuracy requires exact agreement with ground truth,
while relaxed accuracy credits cautious explanations under ambiguity.}
\end{table}
""")

    print("✅ Semantic explanation benchmark complete")
    print(json.dumps(results, indent=2))


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="semantic_eval_results.json")
    parser.add_argument("merge_gt_json", help="merge_with_gt.json")
    parser.add_argument("--out_json", default="semantic_benchmark_gt.json")
    parser.add_argument("--out_latex", default="semantic_benchmark_gt.tex")
    args = parser.parse_args()

    run_benchmark(
        args.input_json,
        args.merge_gt_json,
        args.out_json,
        args.out_latex
    )
