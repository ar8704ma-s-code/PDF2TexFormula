import os
import json

SRC_DIR = "data/aligned_dataset_filtered_2"
OUT_DIR = "data/train_data_2"

os.makedirs(OUT_DIR, exist_ok=True)

with_gt_path = os.path.join(OUT_DIR, "with_gt.jsonl")
noise_path   = os.path.join(OUT_DIR, "noise.jsonl")
merged_path  = os.path.join(OUT_DIR, "train_data_2.jsonl")

with_gt_f = open(with_gt_path, "w", encoding="utf8")
noise_f   = open(noise_path, "w", encoding="utf8")
merged_f  = open(merged_path, "w", encoding="utf8")

VALID_TASKS = {"with_gt", "noise"}

count_gt = 0
count_noise = 0
count_total = 0

for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".jsonl"):
        continue

    path = os.path.join(SRC_DIR, fname)
    print(f"[Reading] {path}")

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)
            task = sample.get("task", "").strip()

            if task not in VALID_TASKS:
                # 忽略所有其他任务
                continue

            # 写入 train_data_2.jsonl
            merged_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count_total += 1

            # 分类写入
            if task == "with_gt":
                with_gt_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count_gt += 1
            else:  # task == "noise"
                noise_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count_noise += 1

with_gt_f.close()
noise_f.close()
merged_f.close()

print("====================================")
print("Extraction Completed ✔")
print("with_gt samples :", count_gt)
print("noise samples   :", count_noise)
print("total samples   :", count_total)
print("Output saved to :", OUT_DIR)
print("====================================")
