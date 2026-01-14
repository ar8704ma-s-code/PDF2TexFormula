import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_loader import load_jsonl
from utils.evaluation import evaluate

MODEL_DIR = "models/lora_formula_checker"
TESTSET = "dataset/formula_dataset.jsonl"


def run_inference(model, tokenizer, formula):

    prompt = (
        "Evaluate this formula.\n"
        f"Formula: {formula}\n"
        "Answer in JSON with {correct, reason, suggestion}"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)

    try:
        js = json.loads(txt.split("{")[-1].split("}")[0] + "}")
    except:
        js = {"correct": False}

    return js


def main():
    print("Loading model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto")

    print("Loading testset…")
    data = load_jsonl(TESTSET)
    gold = [{"correct": d["label_correct"]} for d in data]

    preds = []
    for item in data[:300]:  # 300 条快速评估
        preds.append(run_inference(model, tokenizer, item["input"]))

    print("Evaluation result:")
    print(evaluate(preds, gold[:300]))


if __name__ == "__main__":
    main()
