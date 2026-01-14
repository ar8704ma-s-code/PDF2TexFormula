# build_dataset_A.py
import json
from pathlib import Path

INPUT_DIR = Path("data/aligned_dataset_refined")
OUTPUT_DIR = Path("data/train_A_withgt_noise")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_TASKS = {"with_gt", "noise"}

def main():
    files = sorted(INPUT_DIR.glob("*.jsonl"))
    print(f"Found {len(files)} files.")

    total_kept = 0
    for f in files:
        out_path = OUTPUT_DIR / f.name
        with f.open("r", encoding="utf-8") as fin, \
             out_path.open("w", encoding="utf-8") as fout:

            for line in fin:
                rec = json.loads(line)
                if rec["task"] in KEEP_TASKS:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_kept += 1

    print(f"Done. Kept {total_kept} clean samples (with_gt + noise).")

if __name__ == "__main__":
    main()
