import json
import re
import difflib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


# ==================================================
# Directories
# ==================================================
GT_DIR = Path("data/arxiv_math_test/gt_cleaned_2")
OCR_DIR = Path("data/Pix2tex_test_1")
OUTPUT_DIR = Path("data/aligned_dataset_test_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# Page sorting utilities
# ==================================================
def page_index(page: str) -> int:
    """
    Extract page index from names like 'page_0032'.
    Falls back to 0 if parsing fails.
    """
    try:
        return int(re.sub(r"[^\d]", "", page))
    except Exception:
        return 0


def sort_key(ocr_item: dict):
    """
    Sort OCR records by (page_index, y, x) so that
    page order and reading order are respected.
    """
    bbox = ocr_item.get("bbox", [0.0, 0.0, 0.0, 0.0])
    return (
        page_index(ocr_item.get("page", "")),
        float(bbox[1]),
        float(bbox[0]),
    )


# ==================================================
# Multi-level normalization for LaTeX similarity
# ==================================================
def semantic_normalize(latex: str) -> str:
    """
    Normalize LaTeX into a semantically more comparable form:
      - Remove spaces
      - Normalize fractions
      - Remove \left / \right
      - Strip style commands, keep inner content
      - Remove spacing commands
    """
    if not latex:
        return ""

    s = latex.replace(" ", "")

    # Normalize fractions
    s = re.sub(r"\\dfrac\b", r"\\frac", s)
    s = re.sub(r"\\tfrac\b", r"\\frac", s)
    s = re.sub(r"\{([^}]+)\s*\\over\s*([^}]+)\}", r"\\frac{\1}{\2}", s)

    # Remove \left and \right (but keep brackets)
    s = re.sub(r"\\left", "", s)
    s = re.sub(r"\\right", "", s)

    # Remove style commands, keep their content
    style_cmds = [
        r"\\mathbf", r"\\boldsymbol", r"\\mathrm", r"\\textbf",
        r"\\mathit", r"\\mathsf", r"\\mathtt", r"\\mathcal", r"\\mathbb"
    ]
    for cmd in style_cmds:
        s = re.sub(cmd + r"\s*\{", "{", s)

    # Remove micro spacing commands
    s = re.sub(r"\\,|\\:|\\;|\\!|\\quad|\\qquad", "", s)

    return s


def structural_normalize(latex: str) -> str:
    """
    Extract a structural representation of the formula:
      - Semantic normalization
      - Remove digits
      - Keep only math-ish tokens: letters, commands, operators, brackets
    """
    s = semantic_normalize(latex)

    # Remove all digits
    s = re.sub(r"\d+", "", s)

    # Extract relevant math tokens
    tokens = re.findall(r"[a-zA-Z]|\\[a-zA-Z]+|[+\-*/=^_(){}\[\]]", s)
    return "".join(tokens)


# ==================================================
# Multi-level similarity
# ==================================================
def multi_level_similarity(a: str, b: str) -> dict:
    """
    Compute several similarity scores between LaTeX strings:
      - strict string similarity
      - semantic similarity (after semantic_normalize)
      - structural similarity (after structural_normalize)
      - element overlap (shared math commands/operators)
      - combined weighted score
    """
    # Strict similarity on raw strings
    strict = difflib.SequenceMatcher(None, a, b).ratio()

    # Semantic similarity
    sem_a = semantic_normalize(a)
    sem_b = semantic_normalize(b)
    semantic = difflib.SequenceMatcher(None, sem_a, sem_b).ratio()

    # Structural similarity
    struct_a = structural_normalize(a)
    struct_b = structural_normalize(b)
    structural = difflib.SequenceMatcher(None, struct_a, struct_b).ratio()

    # Overlap between math elements (commands / operators)
    elems_a = set(re.findall(r"\\[a-zA-Z]+|[=+\-*/^_()]", sem_a))
    elems_b = set(re.findall(r"\\[a-zA-Z]+|[=+\-*/^_()]", sem_b))
    if elems_a or elems_b:
        elem_overlap = len(elems_a & elems_b) / len(elems_a | elems_b)
    else:
        elem_overlap = 0.0

    # Combined score (hand-tuned weights)
    combined = (
        0.4 * semantic +
        0.3 * strict +
        0.2 * structural +
        0.1 * elem_overlap
    )

    return {
        "strict": strict,
        "semantic": semantic,
        "structural": structural,
        "element_overlap": elem_overlap,
        "combined": combined
    }


# ==================================================
# Noise detection (aggressive for numeric labels)
# ==================================================
NOISE_PATTERNS = [
    r"^\(?\d+(\.\d+)*\)?$",              # (12), 12, 3.5
    r"^\[?\d+(\.\d+)*\]?$",              # [3], [2.1]
    r"^\\textbf\{?\(?\d+(\.\d+)*\)?\}?$",  # \textbf{(1.3)}
    r"^\\label\{[^}]+\}$",
    r"^\\ref\{[^}]+\}$",
    r"^\\eqref\{[^}]+\}$",
]


def is_noise(text: str) -> bool:
    """
    Decide if an OCR segment is "pure noise":
      - Almost empty
      - Pure numeric labels / references
      - Very short, with no math commands or operators
    """
    if not text or not text.strip():
        return True

    s = text.strip()

    # Single character is rarely a meaningful formula
    if len(s) <= 1:
        return True

    # Numeric label patterns
    for p in NOISE_PATTERNS:
        if re.fullmatch(p, s):
            return True

    # Short text with no math content
    if len(s) < 6 and not re.search(r"[\\=+\-*/_^]", s):
        return True

    return False


# ==================================================
# Record creation helpers
# ==================================================
def create_with_gt_record(ocr: dict, gt: dict, scores: dict) -> dict:
    """
    Create a record for an OCR formula that is matched to a GT formula.
    """
    return {
        "task": "with_gt",
        "gt_id": gt["gt_id"],
        "gt_latex": gt["latex"],
        "similarity": float(scores["combined"]),
        "strict_similarity": float(scores["strict"]),
        "semantic_similarity": float(scores["semantic"]),
        "structural_similarity": float(scores["structural"]),
        "element_overlap": float(scores["element_overlap"]),
        **ocr,
    }


from typing import Optional, Dict

def create_repair_no_gt_record(ocr: dict, scores: Optional[Dict] = None) -> dict:
    ...

    """
    Create a record for an OCR formula that looks like a formula but
    could not be reliably matched to any GT formula.
    """
    base = {
        "task": "repair_no_gt",
        "gt_id": None,
        "gt_latex": None,
        "similarity": -1.0,
        "strict_similarity": -1.0,
        "semantic_similarity": -1.0,
        "structural_similarity": -1.0,
        "element_overlap": -1.0,
        **ocr,
    }

    if scores is not None:
        base.update(
            similarity=float(scores["combined"]),
            strict_similarity=float(scores["strict"]),
            semantic_similarity=float(scores["semantic"]),
            structural_similarity=float(scores["structural"]),
            element_overlap=float(scores["element_overlap"]),
        )

    return base


def create_noise_record(ocr: dict) -> dict:
    """
    Create a record for pure noise segments.
    """
    return {
        "task": "noise",
        "gt_id": None,
        "gt_latex": None,
        "similarity": -1.0,
        "strict_similarity": -1.0,
        "semantic_similarity": -1.0,
        "structural_similarity": -1.0,
        "element_overlap": -1.0,
        **ocr,
    }


# ==================================================
# Greedy matching (first stage)
# ==================================================
def greedy_match(ocr_sorted: list[dict], gt_list: list[dict], match_threshold: float = 0.55):
    """
    Greedy matching:
      - For each OCR formula (in reading order), find the best unmatched GT.
      - If best combined score >= threshold -> with_gt
      - Else -> repair_no_gt
      - Pure noise is filtered by is_noise() first.
    """
    used_gt_ids = set()
    aligned = []

    for ocr in ocr_sorted:
        raw = ocr["raw_ocr"]

        # Clear numeric labels / trivial garbage
        if is_noise(raw):
            aligned.append(create_noise_record(ocr))
            continue

        best_gt = None
        best_score = 0.0
        best_scores = None

        for gt in gt_list:
            if gt["gt_id"] in used_gt_ids:
                continue
            scores = multi_level_similarity(raw, gt["latex"])
            if scores["combined"] > best_score:
                best_score = scores["combined"]
                best_gt = gt
                best_scores = scores

        if best_gt is not None and best_score >= match_threshold:
            # Reliable match found
            used_gt_ids.add(best_gt["gt_id"])
            aligned.append(create_with_gt_record(ocr, best_gt, best_scores))
        else:
            # Looks like formula (since not noise), but no good GT match
            aligned.append(create_repair_no_gt_record(ocr, best_scores))

    return aligned


# ==================================================
# Hungarian matching (fallback)
# ==================================================
def hungarian_match(ocr_sorted: list[dict], gt_list: list[dict], match_threshold: float = 0.55):
    """
    Hungarian optimal assignment (global) matching:
      - Build similarity matrix between OCR and GT.
      - Use linear_sum_assignment to get best global matching.
      - If matched score >= threshold -> with_gt
      - Else -> repair_no_gt
      - Pure noise is still decided by is_noise().
    """
    if not ocr_sorted or not gt_list:
        return []

    n_ocr = len(ocr_sorted)
    n_gt = len(gt_list)

    sim_matrix = np.zeros((n_ocr, n_gt), dtype=float)
    sim_store: list[list[dict]] = [[None] * n_gt for _ in range(n_ocr)]

    for i, ocr in enumerate(ocr_sorted):
        raw = ocr["raw_ocr"]
        for j, gt in enumerate(gt_list):
            scores = multi_level_similarity(raw, gt["latex"])
            sim_matrix[i, j] = scores["combined"]
            sim_store[i][j] = scores

    # Hungarian algorithm maximizes similarity => minimize negative
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)
    matched_pairs = set(zip(row_ind, col_ind))

    aligned = []

    for i, ocr in enumerate(ocr_sorted):
        raw = ocr["raw_ocr"]

        # First check for pure noise
        if is_noise(raw):
            aligned.append(create_noise_record(ocr))
            continue

        chosen_scores = None
        chosen_gt = None

        # Check if OCR i is in the matched pairs
        for j, gt in enumerate(gt_list):
            if (i, j) in matched_pairs:
                scores = sim_store[i][j]
                chosen_scores = scores
                chosen_gt = gt
                break

        if chosen_gt is not None and chosen_scores["combined"] >= match_threshold:
            aligned.append(create_with_gt_record(ocr, chosen_gt, chosen_scores))
        else:
            # No reliable GT match, but still a formula-like string
            aligned.append(create_repair_no_gt_record(ocr, chosen_scores))

    return aligned


# ==================================================
# Hybrid alignment: greedy + fallback
# ==================================================
def align_ocr_to_gt(ocr_list: list[dict], gt_list: list[dict]) -> list[dict]:
    """
    Hybrid alignment strategy:
      1. Sort OCR by reading order.
      2. Run greedy matching.
      3. If greedy coverage (with_gt / total GT) is too low, fallback to Hungarian.
      4. Return list of aligned records with task in {with_gt, repair_no_gt, noise}.
    """
    ocr_sorted = sorted(ocr_list, key=sort_key)

    # Stage 1: greedy
    greedy_result = greedy_match(ocr_sorted, gt_list)
    with_gt_count = sum(1 for r in greedy_result if r["task"] == "with_gt")
    coverage = with_gt_count / len(gt_list) if gt_list else 0.0

    # If coverage is very poor, try global Hungarian matching
    if coverage < 0.35:
        return hungarian_match(ocr_sorted, gt_list)

    return greedy_result


# ==================================================
# Loaders
# ==================================================
def load_gt(paper_id: str) -> list[dict]:
    """
    Load GT formulas for a given paper.
    Each line in gt_cleaned is: { "paper_id", "gt_id", "latex_gt" }.
    """
    path = GT_DIR / f"{paper_id}.jsonl"
    if not path.exists():
        return []

    formulas = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            formulas.append({
                "gt_id": obj["gt_id"],
                "latex": obj["latex_gt"],
            })
    return formulas


def load_ocr(paper_id: str) -> list[dict]:
    """
    Load OCR formulas for a given paper from overall_ocr_results.json.
    """
    path = OCR_DIR / paper_id / "overall_ocr_results.json"
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for page, page_data in data.items():
        for f in page_data.get("formulas", []):
            items.append({
                "page": page,
                "bbox": f.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                "raw_ocr": f.get("raw_latex", "") or "",
            })
    return items


# ==================================================
# Per-paper processing
# ==================================================
def process_paper(paper_id: str) -> tuple[int, int, int, int]:
    """
    Align all OCR formulas of a paper to its GT formulas.
    Returns:
        (with_gt_count, repair_no_gt_count, noise_count, total_records)
    """
    gt_list = load_gt(paper_id)
    ocr_list = load_ocr(paper_id)

    if not gt_list or not ocr_list:
        return 0, 0, 0, 0

    aligned = align_ocr_to_gt(ocr_list, gt_list)

    # Inject paper_id into each record
    for rec in aligned:
        rec["paper_id"] = paper_id

    # Save aligned dataset for this paper
    out_path = OUTPUT_DIR / f"{paper_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in aligned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Statistics per paper
    with_gt_count = sum(1 for r in aligned if r["task"] == "with_gt")
    repair_no_gt_count = sum(1 for r in aligned if r["task"] == "repair_no_gt")
    noise_count = sum(1 for r in aligned if r["task"] == "noise")
    total_records = len(aligned)

    print(
        f"{paper_id}: total={total_records}, "
        f"with_gt={with_gt_count}, "
        f"repair_no_gt={repair_no_gt_count}, "
        f"noise={noise_count}"
    )

    return with_gt_count, repair_no_gt_count, noise_count, total_records


# ==================================================
# Batch processing and global summary
# ==================================================
def batch_align():
    """
    Align all papers that have both GT and OCR.
    At the end, print a global summary:
        - number of processed papers
        - total samples
        - count of with_gt / repair_no_gt / noise
    """
    gt_papers = {p.stem for p in GT_DIR.glob("*.jsonl")}
    ocr_papers = {d.name for d in OCR_DIR.iterdir() if d.is_dir()}
    papers = sorted(gt_papers & ocr_papers)

    total_with_gt = 0
    total_repair_no_gt = 0
    total_noise = 0
    total_samples = 0
    processed_papers = 0

    for pid in tqdm(papers, desc="Aligning papers"):
        with_gt_count, repair_no_gt_count, noise_count, n_records = process_paper(pid)
        if n_records == 0:
            continue

        processed_papers += 1
        total_with_gt += with_gt_count
        total_repair_no_gt += repair_no_gt_count
        total_noise += noise_count
        total_samples += n_records

    # Global summary
    print("\n===== Aligned Summary =====")
    print(f"Processed papers : {processed_papers}")
    print(f"Total samples    : {total_samples}\n")

    print(f"with_gt          : {total_with_gt}")
    print(f"repair_no_gt     : {total_repair_no_gt}")
    print(f"noise            : {total_noise}\n")

    if total_samples > 0:
        print("========== RATIO ==========")
        print(f"with_gt          : {total_with_gt / total_samples:.4f}")
        print(f"repair_no_gt     : {total_repair_no_gt / total_samples:.4f}")
        print(f"noise            : {total_noise / total_samples:.4f}")

    print("\nOutput directory :", OUTPUT_DIR)


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    batch_align()
