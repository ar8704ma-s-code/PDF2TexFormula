# src/pipeline/pipeline_raw.py

import sys
import json
import re
from pathlib import Path

from detection_ocr import DetectionOCR


# ============================================================
# Output directory (RAW baseline)
# ============================================================
def get_output_directory(pdf_path: Path) -> Path:
    pdf_name = pdf_path.stem
    out_root = Path("output_raw")
    out_root.mkdir(exist_ok=True)

    pdf_output_dir = out_root / pdf_name
    pdf_output_dir.mkdir(exist_ok=True)

    print(f"[RAW] PDF = {pdf_path.name}")
    print(f"[RAW] Output directory = {pdf_output_dir}")
    return pdf_output_dir


# ============================================================
# Step 1 — OCR only
# ============================================================
def run_ocr(pdf_path: Path, output_dir: Path):
    detector = DetectionOCR()

    results = detector.process_pdf(pdf_path, str(output_dir))

    json_path = output_dir / "overall_ocr_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"[RAW-OCR] Saved: {json_path}")
    return results


# ============================================================
# Step 2 — Remove bbox PNGs
# ============================================================
def remove_bbox_pngs(output_dir: Path):
    removed = 0
    for p in output_dir.glob("page_*_bbox.png"):
        p.unlink(missing_ok=True)
        removed += 1
    print(f"[RAW] Removed {removed} bbox PNGs.")


# ============================================================
# Step 3 — Remove pages/ folder
# ============================================================
def remove_pages_dir(output_dir: Path):
    pages_dir = output_dir / "pages"
    if pages_dir.exists():
        for f in pages_dir.glob("*"):
            f.unlink()
        pages_dir.rmdir()
        print("[RAW] Removed pages/ directory.")
    else:
        print("[RAW] No pages/ directory found.")


# ============================================================
# Step 4 — Remove page_*_results.json
# ============================================================
def remove_single_page_json(output_dir: Path):
    removed = 0
    for p in output_dir.glob("page_*_results.json"):
        p.unlink(missing_ok=True)
        removed += 1
    print(f"[RAW] Removed {removed} page_*_results.json files.")


# ============================================================
# Step 5 — Remove empty TEX files (without equations)
# ============================================================
def remove_empty_tex(output_dir: Path):
    removed = 0
    pattern = r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}"

    for tex_path in output_dir.glob("page_*_formulas.tex"):
        content = tex_path.read_text(errors="ignore")
        matches = re.findall(pattern, content, flags=re.S)
        if len(matches) == 0:
            tex_path.unlink()
            removed += 1

    print(f"[RAW] Removed {removed} empty page_*_formulas.tex files.")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_raw.py <pdf_file>")
        return

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF not found → {pdf_path}")
        return

    output_dir = get_output_directory(pdf_path)

    print("\n=== 1) OCR ===")
    run_ocr(pdf_path, output_dir)

    print("\n=== 2) Remove bbox PNGs ===")
    remove_bbox_pngs(output_dir)

    print("\n=== 3) Remove pages directory ===")
    remove_pages_dir(output_dir)

    print("\n=== 4) Remove page_*_results.json ===")
    remove_single_page_json(output_dir)

    print("\n=== 5) Remove empty TEX files ===")
    remove_empty_tex(output_dir)

    print("\n[RAW] Baseline pipeline finished.\n")


if __name__ == "__main__":
    main()
