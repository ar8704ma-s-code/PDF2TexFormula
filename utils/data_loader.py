import json

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_supervised(example):
    return {
        "instruction": "Evaluate formula correctness.",
        "input": example["input"],
        "output": {
            "correct": example["label_correct"],
            "reason": example["label_reason"],
            "suggestion": example.get("suggestion", "")
        }
    }
