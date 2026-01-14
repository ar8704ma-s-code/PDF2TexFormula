import json
import fitz  # PyMuPDF
from pathlib import Path
import shutil
from PIL import Image
import io

ROOT = Path("data/Pix2tex_test_1")
PDF_ROOT = Path("data/arxiv_math_test/pdf")

OUT_IMG = Path("clean_formula_images")
OUT_IMG.mkdir(exist_ok=True)

OUT_PDF = Path("bbox_pdf")
OUT_PDF.mkdir(exist_ok=True)


def crop(pdf, page_num, bbox):
    """裁剪 bbox 区域并返回 Pillow Image"""
    page = pdf[page_num]
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(clip=rect, dpi=200)

    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def concat_vertical(images):
    """按顺序竖向拼接多张公式图像"""
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    canvas = Image.new("RGB", (max(widths), sum(heights)), "white")
    offset = 0
    for img in images:
        canvas.paste(img, (0, offset))
        offset += img.height

    return canvas


def process_paper(paper_dir: Path):
    paper_id = paper_dir.name
    json_file = paper_dir / "overall_ocr_results.json"
    pdf_file = PDF_ROOT / f"{paper_id}.pdf"

    if not json_file.exists() or not pdf_file.exists():
        print(f"❌ Missing {paper_id}")
        return

    print(f"📄 Processing {paper_id}...")
    data = json.loads(json_file.read_text())
    pdf = fitz.open(pdf_file)

    ocrimg_dir = paper_dir / "ocrimg"
    ocrimg_dir.mkdir(exist_ok=True)

    page_images = []

    for page_key, page_info in data.items():
        page_num = int(page_key.replace("page_", "")) - 1
        formulas = page_info.get("formulas", [])

        if not formulas:
            continue

        cropped_imgs = []

        for idx, f in enumerate(formulas, start=1):
            bbox = f["bbox"]
            try:
                img = crop(pdf, page_num, bbox)
                save_path = ocrimg_dir / f"{page_key}_f{idx}.png"
                img.save(save_path)

                cropped_imgs.append(img)

            except Exception as e:
                print(f"❌ Crop fail {paper_id} {page_key} #{idx}: {e}")

        if cropped_imgs:
            merged = concat_vertical(cropped_imgs)
            merged_path = ocrimg_dir / f"{page_key}_bbox.png"
            merged.save(merged_path)

            page_images.append(merged)

    # 输出 PDF
    if page_images:
        pdf_save = OUT_PDF / f"{paper_id}.pdf"
        page_images[0].save(pdf_save, save_all=True, append_images=page_images[1:])
        print(f"✔ Saved PDF: {pdf_save}")

    pdf.close()


def main():
    for paper_dir in ROOT.iterdir():
        if paper_dir.is_dir():
            process_paper(paper_dir)

    print("\n✨ Done. Formula images + bbox PDFs generated.")


if __name__ == "__main__":
    main()
