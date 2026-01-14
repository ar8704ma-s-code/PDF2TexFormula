import json
from pathlib import Path

IN_DIR = Path("data/train_A_withgt_noise")
OUT_DIR = Path("data/train_A_withgt_noise_cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_file(in_path: Path, out_path: Path):
    cleaned = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            task = record.get("task", "")
            gt = record.get("gt_latex", None)

            if task == "noise":
                # 强制设置 noise 的 ground truth 为字符串 "noise"
                record["gt_latex"] = "noise"

            elif task == "with_gt":
                # 保留正常数学
                if not isinstance(gt, str) or gt.strip() == "":
                    print(f"[Warning] Empty gt_latex in: {in_path.name}")

            # 严格保证输出可训练
            if "gt_latex" not in record or record["gt_latex"] is None:
                print(f"[Fix] Missing gt_latex in {in_path.name}, setting to 'noise'")
                record["gt_latex"] = "noise"

            cleaned.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        for r in cleaned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    files = sorted(IN_DIR.glob("*.jsonl"))
    print(f"Found {len(files)} files to clean.\n")

    for file in files:
        out_path = OUT_DIR / file.name
        print(f"Cleaning: {file.name}")
        clean_file(file, out_path)

    print("\n=== DONE! Cleaned files saved to data/train_A_withgt_noise_cleaned ===")

if __name__ == "__main__":
    main()
