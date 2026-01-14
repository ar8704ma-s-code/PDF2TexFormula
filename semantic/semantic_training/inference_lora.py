import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE = "Qwen/Qwen1.5-0.5B" 
ADAPTER = "/home/baiyinyou/workspace/SmolLatexFormula/models/Qwen_formula_lora"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()
    return tokenizer, model


def check_formula(text: str):
    tokenizer, model = load_model()

    prompt = (
        "你是一名公式审核助手，请判断以下公式是否正确：\n"
        f"公式：{text}\n"
        "回答格式：正确性：正确/错误\n原因：一句话解释\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
        )

    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    check_formula("H2 + O2 -> H2O")
