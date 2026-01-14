import torch
from PIL import Image, ImageDraw
from typing import List, Dict, Any
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10
from pix2tex import cli as pix2tex
from pathlib import Path
import json
import fitz



class DetectionOCR:

    def __init__(self, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[DetectionOCR] Using {self.device}")

        # Load YOLO model
        ckpt = "models/yolo.pt"
        print("[DetectionOCR] Loading YOLO checkpoint...")
        self.yolo = YOLOv10(ckpt).to(self.device)
        self.names = self.yolo.names  # ← FIX

        # Load Pix2TeX OCR
        print("[DetectionOCR] Loading LatexOCR...")
        self.pix2tex = pix2tex.LatexOCR()

    # ------------------------------------------------------------
    # 1. DETECTION
    # ------------------------------------------------------------

    def detect_formula_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect formula regions using YOLO
        """

        det = self.yolo.predict(
            image_path,
            imgsz=1024,
            conf=0.25,
            device=self.device,
            verbose=False
        )[0]

        xyxy = det.boxes.xyxy.cpu().numpy()
        cls_ids = det.boxes.cls.cpu().numpy().astype(int)
        scores = det.boxes.conf.cpu().numpy()

        results = []
        for i, cid in enumerate(cls_ids):
            label = self.names[cid]
            if "formula" in label.lower():
                x1, y1, x2, y2 = xyxy[i]
                results.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "label": label,
                    "score": float(scores[i])
                })
        return results

    # ------------------------------------------------------------
    # 2. OCR
    # ------------------------------------------------------------

    def extract_latex_from_region(self, page_img: Image.Image, region) -> str:
        x1, y1, x2, y2 = map(int, region["bbox"])
        crop = page_img.crop((x1, y1, x2, y2))

        try:
            result = self.pix2tex(crop)
            if isinstance(result, dict):
                return result.get("latex", "")
            return str(result)
        except Exception as e:
            print("[OCR] Error:", e)
            return ""
        
    # ------------------------------------------------------------
    # 3. DRAW BBOX
    # ------------------------------------------------------------

    def draw_bounding_boxes(self, image_path, regions, output_path):
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for i, region in enumerate(regions):
            x1, y1, x2, y2 = map(int, region["bbox"])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, max(0, y1-20)), f"Formula {i+1}", fill="red")

        img.save(output_path)
        print(f"[BBox] Saved:", output_path)

    # ------------------------------------------------------------
    # 4. SAVE TEX
    # ------------------------------------------------------------

    def save_latex_to_tex(self, regions, output_path):

        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath, amssymb}",
            r"\begin{document}"
        ]

        for i, r in enumerate(regions):
            lines.append(r"\begin{equation*}")
            lines.append(r.get("raw_latex", ""))
            lines.append(r"\end{equation*}")
            lines.append("")

        lines.append(r"\end{document}")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        print("[TeX] Saved:", output_path)

    # ------------------------------------------------------------
    # 5. SAVE JSON
    # ------------------------------------------------------------

    def save_results_to_json(self, regions, output_path):
        Path(output_path).write_text(
            json.dumps({"formulas": regions, "total": len(regions)}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print("[JSON] Saved:", output_path)

    # ------------------------------------------------------------
    # 6. PROCESS IMAGE
    # ------------------------------------------------------------

    def process_image(self, image_path, output_dir):

        Path(output_dir).mkdir(exist_ok=True)
        img_name = Path(image_path).stem

        page = Image.open(image_path).convert("RGB")
        regions = self.detect_formula_regions(image_path)

        results = []
        for region in regions:
            latex = self.extract_latex_from_region(page, region)
            region["raw_latex"] = latex
            results.append(region)

        # Save bbox image
        bbox_path = f"{output_dir}/{img_name}_bbox.png"
        self.draw_bounding_boxes(image_path, results, bbox_path)

        # Save tex
        tex_path = f"{output_dir}/{img_name}_formulas.tex"
        self.save_latex_to_tex(results, tex_path)

        # Save json
        json_path = f"{output_dir}/{img_name}_results.json"
        self.save_results_to_json(results, json_path)

        return results

    # ------------------------------------------------------------
    # 7. PROCESS PDF
    # ------------------------------------------------------------

    def process_pdf(self, pdf_path, output_dir="output"):
        Path(output_dir).mkdir(exist_ok=True)
        tmp_dir = Path(output_dir) / "pages"
        tmp_dir.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)
        all_results = {}

        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=300)

            img_file = tmp_dir / f"page_{i+1}.png"
            pix.save(str(img_file))

            print(f"\n[PDF] Page {i+1}")
            page_results = self.process_image(str(img_file), output_dir)

            all_results[f"page_{i+1}"] = {
                "total": len(page_results),
                "formulas": page_results
            }

        overall_json = Path(output_dir) / "overall_ocr_results.json"
        overall_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), "utf-8")

        print("\n== Done ==")
        print("Output:", output_dir)
        return all_results

