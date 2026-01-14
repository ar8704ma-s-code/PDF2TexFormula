import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from detection_ocr import DetectionOCR
from repair_head import RepairHead, is_structurally_valid_latex


# ============================================================
# CONFIG
# ============================================================
LORA_DIR = "src/repair/repair_training/Repair_improved_1.5B_v1"


MATH_ENV_PATTERNS = [
    r"\\begin\{align\*?\}.*?\\end\{align\*?\}",
    r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
    r"\\begin\{align\}.*?\\end\{align\}",
    r"\\begin\{equation\}.*?\\end\{equation\}",
]


# ============================================================
# Complete Pipeline
# ============================================================
class CompletePipeline:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.detector = DetectionOCR(device=device)

    # ============================================================
    # Step 1 — OCR
    # ============================================================
    def get_output_directory(self, pdf_path: Path) -> Path:
        out_root = Path("output_1.5B")
        out_root.mkdir(exist_ok=True)

        pdf_output_dir = out_root / pdf_path.stem
        pdf_output_dir.mkdir(exist_ok=True)

        return pdf_output_dir

    def run_ocr(self, pdf_path: Path, output_dir: Path):
        results = self.detector.process_pdf(pdf_path, str(output_dir))
        json_path = output_dir / "overall_ocr_results.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    # ============================================================
    # Step 2 — Merge bbox → pdf_bbox.pdf (按页面顺序合并)
    # ============================================================
    def merge_bbox_pngs_to_pdf(self, output_dir: Path):
        png_files = list(output_dir.glob("page_*_bbox.png"))
        if not png_files:
            return

        try:
            # 按照页面数字排序：page_1_bbox.png, page_2_bbox.png, ...
            def get_page_number(filename):
                match = re.search(r'page_(\d+)_bbox\.png', filename.name)
                return int(match.group(1)) if match else 0
            
            png_files_sorted = sorted(png_files, key=get_page_number)
            
            images = []
            for png_file in png_files_sorted:
                try:
                    img = Image.open(png_file).convert("RGB")
                    images.append(img)
                    print(f"   📄 添加页面 {get_page_number(png_file)} 到合并PDF")
                except Exception as e:
                    print(f"   ⚠️ 无法打开图片 {png_file.name}: {e}")

            if not images:
                return

            output_pdf = output_dir / "pdf_bbox.pdf"
            images[0].save(
                output_pdf, 
                save_all=True, 
                append_images=images[1:],
                resolution=100.0
            )

            # Remove original PNGs
            for p in png_files:
                p.unlink(missing_ok=True)
            
            print(f"   🖼️  按顺序合并 {len(images)} 个 bbox PNG → pdf_bbox.pdf")
            
        except Exception as e:
            print(f"   ⚠️  bbox 合并失败: {e}")
            import traceback
            traceback.print_exc()

    # ============================================================
    # Step 3 — Remove temp files
    # ============================================================
    def remove_pages_dir(self, output_dir: Path):
        pages_dir = output_dir / "pages"
        if pages_dir.exists():
            for f in pages_dir.glob("*"):
                f.unlink()
            pages_dir.rmdir()

    def remove_single_page_json(self, output_dir: Path):
        for p in output_dir.glob("page_*_results.json"):
            p.unlink(missing_ok=True)

    def remove_empty_tex(self, output_dir: Path):
        pattern = r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}"
        removed_count = 0
        for tex_path in output_dir.glob("page_*_formulas.tex"):
            content = tex_path.read_text(errors="ignore")
            if not re.findall(pattern, content, flags=re.S):
                tex_path.unlink(missing_ok=True)
                removed_count += 1
        if removed_count > 0:
            print(f"   🗑️  删除 {removed_count} 个空 TeX 文件")

    def cleanup_intermediate_files(self, output_dir: Path):
        print("🧹 清理中间文件...")
        self.merge_bbox_pngs_to_pdf(output_dir)
        self.remove_pages_dir(output_dir)
        self.remove_single_page_json(output_dir)
        self.remove_empty_tex(output_dir)

    # ============================================================
    # Step 4 — Extract formula blocks
    # ============================================================
    def collect_all_tex_files(self, output_dir: Path) -> str:
        all_content = []
        tex_files = sorted(output_dir.glob("page_*_formulas.tex"))
        print(f"   📄 找到 {len(tex_files)} 个 TeX 文件")
        
        for tex_file in tex_files:
            try:
                content = tex_file.read_text(encoding="utf-8", errors="ignore")
                all_content.append(content)
            except Exception as e:
                print(f"   ⚠️ 无法读取 {tex_file.name}: {e}")
        return "\n".join(all_content)

    def extract_math_blocks(self, text: str):
        text = re.sub(r"\\documentclass.*?\\begin\{document\}", "", text, flags=re.S)
        text = re.sub(r"\\end\{document\}", "", text)
        text = text.replace("noise", "")

        blocks = []
        for pat in MATH_ENV_PATTERNS:
            found = re.findall(pat, text, flags=re.S)
            blocks.extend([f.strip() for f in found])

        return blocks

    def normalize_block(self, block: str):
        block = re.sub(r"\\end\{align\*?\}.*", r"\\end{align*}", block, flags=re.S)
        block = re.sub(r"\\end\{equation\*?\}.*", r"\\end{equation*}", block, flags=re.S)
        return block.strip()

    # ============================================================
    # Step 5 — Repair via LoRA
    # ============================================================
    def repair_formulas(self, blocks, repair_head: RepairHead):
        repaired = []
        for b in blocks:
            out = repair_head.repair(b)
            if out.lower() != "noise" and is_structurally_valid_latex(out):
                repaired.append(out)
        return repaired

    # ============================================================
    # Step 6 — Write final LaTeX file
    # ============================================================
    def write_final_latex(self, output_dir: Path, repaired_list):
        output_tex = output_dir / "final_formulas.tex"

        preamble = r"""
\documentclass{article}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{bm}
\begin{document}
"""

        ending = r"""
\end{document}
"""

        with output_tex.open("w", encoding="utf-8") as f:
            f.write(preamble)

            for eq in repaired_list:
                content = re.sub(r"\\begin\{.*?\}", "", eq)
                content = re.sub(r"\\end\{.*?\}", "", content)
                content = content.strip()

                if content:
                    f.write("\\begin{equation}\n")
                    f.write(content + "\n")
                    f.write("\\end{equation}\n\n")

            f.write(ending)

        return output_tex

    # ============================================================
    # Main Process
    # ============================================================
    def process_pdf(self, pdf_path: Path):
        output_dir = self.get_output_directory(pdf_path)

        self.run_ocr(pdf_path, output_dir)
        self.cleanup_intermediate_files(output_dir)

        all_text = self.collect_all_tex_files(output_dir)
        blocks = [self.normalize_block(b) for b in self.extract_math_blocks(all_text)]

        repair_head = RepairHead(lora_dir=LORA_DIR)
        repaired_list = self.repair_formulas(blocks, repair_head)

        final_tex = self.write_final_latex(output_dir, repaired_list)
        print(f"FINISHED → {final_tex}")
        return True


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline_complete.py <PDF>")
        return

    pdf_path = Path(sys.argv[1])
    pipeline = CompletePipeline(device="cuda:0")
    pipeline.process_pdf(pdf_path)


if __name__ == "__main__":
    main()