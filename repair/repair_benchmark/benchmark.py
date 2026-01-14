# python src/benchmark/benchmark.py <paper_id>

import json
import re
import tarfile
import tempfile
from pathlib import Path
import difflib
import numpy as np
import sys


class BenchmarkSingle:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.gt_tar = Path("data/arxiv_math/tex") / f"{paper_id}.tar.gz"
        self.pred_dir = Path("output") / paper_id

    # ============================
    # Extract GT formulas (from tar.gz)
    # ============================
    def extract_gt(self):
        if not self.gt_tar.exists():
            print(f"[GT] Missing GT tar.gz: {self.gt_tar}")
            return []

        formulas = []
        patterns = [
            r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}",
            r"\\begin\{equation\}(.*?)\\end\{equation\}",
            r"\\begin\{align\*\}(.*?)\\end\{align\*\}",
            r"\\begin\{align\}(.*?)\\end\{align\}",
            r"\$\$(.*?)\$\$",
            r"\\\[(.*?)\\\]",
        ]

        try:
            with tarfile.open(self.gt_tar, "r:gz") as tar:
                with tempfile.TemporaryDirectory() as tmp:
                    tar.extractall(tmp)
                    tex_files = list(Path(tmp).rglob("*.tex"))

                    if not tex_files:
                        print(f"[GT] No TeX files inside: {self.gt_tar}")
                        return []

                    for tex in tex_files:
                        content = tex.read_text(errors="ignore")
                        for p in patterns:
                            for m in re.findall(p, content, flags=re.S):
                                f = re.sub(r"\s+", " ", m.strip())
                                if len(f) > 5:
                                    formulas.append(f)

        except Exception as e:
            print("[GT] Extraction error:", e)

        return formulas

    # ============================
    # Extract prediction formulas (pipeline repaired)
    # ============================
    def extract_pred(self):
        tex_files = list(self.pred_dir.glob("*_repaired.tex"))
        if not tex_files:
            print(f"[Pred] No repaired tex in: {self.pred_dir}")
            return []

        formulas = []
        for tex in tex_files:
            content = tex.read_text(errors="ignore")
            matches = re.findall(
                r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}",
                content,
                flags=re.S
            )
            for m in matches:
                f = re.sub(r"\s+", " ", m.strip())
                if len(f) > 5:
                    formulas.append(f)

        return formulas

    # ============================
    # Similarity
    # ============================
    def sim(self, a, b): 
        return difflib.SequenceMatcher(None, a, b).ratio()

    # ============================
    # Full benchmark
    # ============================
    def run(self):
        gt = self.extract_gt()
        pred = self.extract_pred()

        if not gt or not pred:
            print(f"[Benchmark] Missing formulas for {self.paper_id}")
            return None

        similarities = [self.sim(g, p) for g in gt for p in pred]

        thresholds = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
        results = {}

        for t in thresholds:
            tp = 0
            used_gt = set()
            used_pred = set()

            for i, g in enumerate(gt):
                for j, p in enumerate(pred):
                    s = self.sim(g, p)
                    if s >= t and i not in used_gt and j not in used_pred:
                        tp += 1
                        used_gt.add(i)
                        used_pred.add(j)

            prec = tp / len(pred)
            rec = tp / len(gt)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

            results[t] = {
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "matched_pairs": tp,
            }

        best_t = max(results.keys(), key=lambda t: results[t]["f1_score"])

        return {
            "paper_id": self.paper_id,
            "gt_count": len(gt),
            "pred_count": len(pred),
            "similarity_stats": {
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "mean": float(np.mean(similarities)),
                "median": float(np.median(similarities)),
            },
            "threshold_results": results,
            "best_threshold": best_t,
        }


# ============================
# Write Markdown Summary
# ============================
def write_summary_md(result, path: Path):
    gt = result["gt_count"]
    pred = result["pred_count"]
    stats = result["similarity_stats"]

    t = result["best_threshold"]
    best = result["threshold_results"][t]

    lines = [
        f"# Benchmark Summary for {result['paper_id']}\n",
        "## Formula Counts",
        f"- Ground truth: **{gt}**",
        f"- Predictions: **{pred}**\n",
        "## Best Threshold",
        f"- Threshold: **{t}**",
        f"- Precision: **{best['precision']:.4f}**",
        f"- Recall: **{best['recall']:.4f}**",
        f"- F1 Score: **{best['f1_score']:.4f}**\n",
        "## Similarity",
        f"- Mean: **{stats['mean']:.4f}**",
        f"- Median: **{stats['median']:.4f}**",
        f"- Max: **{stats['max']:.4f}**",
        f"- Min: **{stats['min']:.4f}**\n",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Benchmark] Markdown saved → {path}")


# ============================
# Main
# ============================
def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <paper_id>")
        return

    pid = sys.argv[1]
    bench = BenchmarkSingle(pid)
    result = bench.run()

    if result is None:
        return

    out_dir = Path("output") / pid / "benchmark_results"
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"[Benchmark] JSON saved → {json_path}")

    md_path = out_dir / "summary.md"
    write_summary_md(result, md_path)


if __name__ == "__main__":
    main()
