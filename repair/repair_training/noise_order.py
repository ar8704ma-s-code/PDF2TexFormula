import json

in_path = "data/train_data_1/noise.jsonl"
out_path = "data/train_data_1/noise_sorted.jsonl"

samples = []

# === 读取所有行 ===
with open(in_path, 'r', encoding='utf8') as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        raw = obj.get("raw_ocr", "")
        samples.append((len(raw), obj))

# === 按 raw_ocr 的字符串长度排序 ===
samples.sort(key=lambda x: x[0])

# === 输出为新的 jsonl 文件 ===
with open(out_path, 'w', encoding='utf8') as f:
    for _, obj in samples:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("排序完成 →", out_path)
