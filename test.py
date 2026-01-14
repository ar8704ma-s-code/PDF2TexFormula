import json
from pathlib import Path
import torch
from repair_head import RepairHead

# ============================================================
# CONFIG
# ============================================================
INPUT_JSON = Path("output_1.5B/2512.05089v1/overall_ocr_results.json")
OUTPUT_JSON = Path("output_1.5B_v1/test/overall_ocr_repaired.json")
OUTPUT_TEX_DIR = Path("output_1.5B_v1/test/repaired_pages")
OUTPUT_TEX_DIR.mkdir(parents=True, exist_ok=True)

LORA_DIR = "src/repair/repair_training/Repair_improved_1.5B_v1"

# ============================================================
# 模型加载配置（防止弃用警告）
# ============================================================
def load_repair_model(lora_dir):
    """安全加载修复模型，避免弃用警告"""
    # 这里假设 RepairHead 内部使用了 transformers 库
    # 如果需要修改 RepairHead 的初始化方式，可以在这里调整
    
    print(f"🔧 Loading repair model from {lora_dir}...")
    
    # 这里假设 RepairHead 类已经正确处理了 dtype 参数
    # 如果 RepairHead 内部有 torch_dtype 参数，需要修改 RepairHead 类
    repair_head = RepairHead(lora_dir=lora_dir)
    
    return repair_head


# ============================================================
# Helper: 判断是否数学
# ============================================================
def is_reasonable_math(expr: str) -> bool:
    if expr.lower() == "noise":
        return False
    if len(expr) < 3:
        return False
    if len(expr) > 2000:
        return False
    return True


# ============================================================
# Helper: 识别是否含有数学环境
# ============================================================
KNOWN_ENV = [
    "equation", "align", "align*", "gather", "multline",
    "array", "split", "cases"
]

def starts_with_env(expr: str) -> bool:
    expr = expr.strip()
    for env in KNOWN_ENV:
        if expr.startswith(f"\\begin{{{env}}}"):
            return True
    return False

def ends_with_env(expr: str) -> bool:
    expr = expr.strip()
    for env in KNOWN_ENV:
        if expr.endswith(f"\\end{{{env}}}"):
            return True
    return False


# ============================================================
# SAFE WRAPPER: 自动判断是否要包 equation
# ============================================================
def wrap_equation(expr: str) -> str:
    expr = expr.strip()

    # 如果已经是某种环境，保持原样
    if starts_with_env(expr) and ends_with_env(expr):
        return expr

    # 如果已经是 $$...$$，保持原样
    if expr.startswith("$$") and expr.endswith("$$"):
        return expr

    # 如果有 \[...\]，保持原样
    if expr.startswith("\\[") and expr.endswith("\\]"):
        return expr

    # 默认自动包 equation（不会嵌套已有环境）
    return f"\\begin{{equation}}\n{expr}\n\\end{{equation}}"


# ============================================================
# Write TEX page
# ============================================================
def write_page_tex(page_id: str, repaired_list):
    out_path = OUTPUT_TEX_DIR / f"{page_id}_repaired.tex"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{amsmath, amssymb, bm}\n")
        f.write("\\begin{document}\n\n")

        for eq in repaired_list:
            tex_block = wrap_equation(eq)
            f.write(tex_block + "\n\n")

        f.write("\\end{document}\n")

    print(f"💾 TEX saved → {out_path}")


# ============================================================
# 设置 transformers 详细级别（可选）
# ============================================================
import os
# 如果需要查看详细的 transformers 信息，取消下面的注释
# os.environ["TRANSFORMERS_VERBOSITY"] = "info"
# 或者设置为 "error" 来减少输出
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"


# ============================================================
# Main
# ============================================================
def main():
    if not INPUT_JSON.exists():
        print(f"❌ JSON not found: {INPUT_JSON}")
        return

    data = json.loads(INPUT_JSON.read_text())

    print("📘 Loaded overall_ocr_results.json")
    
    # 使用安全的模型加载方式
    repair_head = load_repair_model(LORA_DIR)

    repaired_all = {}

    for page, content in data.items():
        print(f"\n==============================")
        print(f"📄 Processing {page} …")
        print("==============================")

        repaired_list = []
        repaired_all[page] = []

        formulas = content.get("formulas", [])

        for idx, fm in enumerate(formulas):
            raw = fm["raw_latex"]

            print(f"\n----- Formula {idx+1} -----")
            print(f"RAW: {raw[:160]}{'...' if len(raw)>160 else ''}")

            out = repair_head.repair(raw)

            if is_reasonable_math(out):
                print("✔ OK, repaired")
                repaired_list.append(out)
                repaired_all[page].append({"raw": raw, "repaired": out})
            else:
                print("❌ Marked as noise")
                repaired_all[page].append({"raw": raw, "repaired": "noise"})

        # export TEX page file
        write_page_tex(page, repaired_list)

    # save JSON
    OUTPUT_JSON.write_text(json.dumps(repaired_all, indent=2, ensure_ascii=False))
    print(f"\n💾 Repaired JSON saved → {OUTPUT_JSON}")

    print("\n🎉 Done!")


if __name__ == "__main__":
    main()