import json
from pathlib import Path
from repair_head import RepairHead

# ============================================================
# CONFIG
# ============================================================
RAW_ROOT = Path("data/Pix2Tex_raw_dataset/Pix2tex_test_1")
OUTPUT_ROOT = Path("output_7B")
LORA_DIR = "src/repair/repair_training/Repair_improved_7B_v1"

KNOWN_ENV = [
    "equation", "align", "align*", "gather", "multline",
    "array", "split", "cases"
]


# ============================================================
# Utility
# ============================================================
def is_reasonable_math(expr: str) -> bool:
    if expr.lower() == "noise":
        return False
    if len(expr) < 3:
        return False
    if len(expr) > 2000:
        return False
    return True


def starts_with_env(expr: str) -> bool:
    for env in KNOWN_ENV:
        if expr.strip().startswith(f"\\begin{{{env}}}"):
            return True
    return False


def ends_with_env(expr: str) -> bool:
    for env in KNOWN_ENV:
        if expr.strip().endswith(f"\\end{{{env}}}"):
            return True
    return False


def wrap_equation(expr: str) -> str:
    expr = expr.strip()

    if starts_with_env(expr) and ends_with_env(expr):
        return expr
    if expr.startswith("$$") and expr.endswith("$$"):
        return expr
    if expr.startswith("\\[") and expr.endswith("\\]"):
        return expr

    return f"\\begin{{equation}}\n{expr}\n\\end{{equation}}"


def write_page_tex(output_dir: Path, page_id: str, repaired_list):
    out_path = output_dir / f"{page_id}_repaired.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{amsmath, amssymb, bm}\n")
        f.write("\\begin{document}\n\n")

        for eq in repaired_list:
            f.write(wrap_equation(eq) + "\n\n")

        f.write("\\end{document}\n")

    print(f"  💾 TEX saved → {out_path}")


# ============================================================
# Single ID Processing
# ============================================================
def process_single_id(id_name: str, repair_head: RepairHead):
    input_json = RAW_ROOT / id_name / "overall_ocr_results.json"

    if not input_json.exists():
        print(f"⚠️ Missing JSON: {input_json}")
        return

    print(f"\n==============================")
    print(f"📘 Processing ID: {id_name}")
    print("==============================")

    data = json.loads(input_json.read_text())

    # Output directories
    output_dir = OUTPUT_ROOT / id_name
    tex_dir = output_dir / "repaired_pages"
    tex_dir.mkdir(parents=True, exist_ok=True)

    repaired_all = {}

    for page, content in data.items():
        print(f"\n📄 Page: {page}")

        repaired_list = []
        repaired_all[page] = []

        formulas = content.get("formulas", [])

        for idx, fm in enumerate(formulas):
            raw = fm.get("raw_latex", "")

            print(f"  - Formula {idx+1}")
            print(f"    RAW: {raw[:120]}{'...' if len(raw)>120 else ''}")

            out = repair_head.repair(raw)

            if is_reasonable_math(out):
                repaired_list.append(out)
                repaired_all[page].append({"raw": raw, "repaired": out})
                print("    ✔ OK")
            else:
                repaired_all[page].append({"raw": raw, "repaired": "noise"})
                print("    ❌ noise")

        write_page_tex(tex_dir, page, repaired_list)

    # Save JSON
    output_json = output_dir / "overall_ocr_repaired.json"
    output_json.write_text(json.dumps(repaired_all, indent=2, ensure_ascii=False))
    print(f"📦 Saved repaired JSON → {output_json}")


# ============================================================
# Main — batch processing many IDs
# ============================================================
def main():
    print("🔧 Loading repair model …")
    repair_head = RepairHead(lora_dir=LORA_DIR)

    # 遍历所有 id 目录
    ids = sorted([p.name for p in RAW_ROOT.iterdir() if p.is_dir()])

    print(f"📚 Found {len(ids)} IDs to process.")

    for id_name in ids:
        process_single_id(id_name, repair_head)

    print("\n🎉 ALL DONE!")


if __name__ == "__main__":
    main()
