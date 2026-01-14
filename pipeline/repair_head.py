# src/repair/repair_head.py

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path


# ============================================================
# Utility functions
# ============================================================

def is_structurally_valid_latex(expr: str) -> bool:
    s = expr.strip()
    if not s:
        return False

    bal = 0
    for ch in s:
        if ch == "{": bal += 1
        elif ch == "}": bal -= 1
        if bal < 0: return False
    if bal != 0: return False

    if s.count(r"\left") != s.count(r"\right"):
        return False
    
    illegal = [
        r"\\cal\\",
        r"\\tilde\{\\cal",
        r"[^\s]\\mathrm",
        r"\\bf\s+\d",
    ]
    for p in illegal:
        if re.search(p, s): return False

    return True


def is_trivial_noise(expr: str) -> bool:
    s = expr.strip()
    if s.lower() == "noise": return True
    if re.match(r"^\(\s*[\w\.]+\s*\)$", s): return True
    if re.match(r"^[=+\-*/]$", s): return True
    return False


# ============================================================
# RepairHead — Unified version for Local 1.5B + Online 7B
# ============================================================

class RepairHead:
    def __init__(
        self,
        mode="local_1_5B",           # "local_1_5B" or "remote_7B"
        device=None,
        lora_dir="src/repair/repair_training/Repair_improved_1.5B_v1",
        debug=False
    ):
        """
        mode = "local_1_5B"
            → loads models/qwen15b (fastest, offline)
        
        mode = "remote_7B"
            → loads Qwen2.5-Math-7B from HF (NO DOWNLOAD)
            → uses HF's streaming loading
        """

        self.debug = debug

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"\n[RepairHead] Mode = {mode}")
        print(f"[RepairHead] Device = {device}")

        # ---------------------------
        # Load tokenizer + model
        # ---------------------------
        if mode == "local_1_5B":
            self._load_local_1_5B(lora_dir)
        elif mode == "remote_7B":
            self._load_remote_7B()
        else:
            raise ValueError("Unknown mode: choose 'local_1_5B' or 'remote_7B'")

        # ---------------------------
        # Training system prompt
        # ---------------------------
        self.SYSTEM_PROMPT = """
You are a LaTeX formula repair specialist.

Your tasks:

1. If the input is MEANINGFUL mathematics:
   - Repair OCR errors
   - Fix LaTeX syntax and bracket structure
   - Preserve the mathematical meaning
   - Output ONLY the repaired LaTeX

2. If the input is NOISE (labels, references, page numbers, broken garbage):
   - Output exactly: noise

Strict rules:
- Do not invent new math
- Do not add explanations
- Output ONLY repaired LaTeX or "noise".
""".strip()


    # ============================================================
    # 1) LOCAL 1.5B MODEL
    # ============================================================
    def _load_local_1_5B(self, lora_dir):
        local_path = Path("models/qwen15b")
        if not local_path.exists():
            raise FileNotFoundError("❌ Local 1.5B model not found in models/qwen15b")

        print("[RepairHead] Loading LOCAL 1.5B model...")

        self.tokenizer = AutoTokenizer.from_pretrained(local_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=torch.float16,
            device_map={"": self.device},
        )

        # attach LoRA
        if lora_dir and Path(lora_dir).exists():
            print("[RepairHead] Loading LoRA...")
            self.model = PeftModel.from_pretrained(self.model, lora_dir)

        self.model.eval()
        print("✔ Local 1.5B model loaded!")


    # ============================================================
    # 2) REMOTE ONLINE 7B MODEL (NO DOWNLOAD)
    # ============================================================
    def _load_remote_7B(self):
        print("[RepairHead] Loading REMOTE 7B (streaming, no download)...")

        base = "Qwen/Qwen2.5-Math-7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            base, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.float16,
            device_map="auto",  # spread across GPU memory
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        print("✔ Remote 7B loaded (streaming)")


    # ============================================================
    # Prompt builder
    # ============================================================
    def build_prompt(self, expr):
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nInput (OCR output): {expr}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


    # ============================================================
    # Repair pipeline
    # ============================================================
    def repair(self, latex: str) -> str:
        if is_trivial_noise(latex):
            return "noise"
        return self._recursive_repair(latex)


    # ============================================================
    # Recursive repair
    # ============================================================
    def _recursive_repair(self, expr, max_iter=4):
        current = expr
        original = expr

        for i in range(max_iter):
            candidate = self._single_step(current)
            if candidate == "noise":
                return "noise"

            if is_structurally_valid_latex(candidate):
                return candidate

            current = candidate

        return current


    # ============================================================
    # Single step repair
    # ============================================================
    def _single_step(self, latex):
        prompt = self.build_prompt(latex)

        toks = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **toks,
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=False)
        return self._extract(decoded)


    # ============================================================
    # Extract assistant message
    # ============================================================
    def _extract(self, txt):
        found = re.findall(
            r"<\|im_start\|>assistant\s*(.*?)(?:<\|im_end\|>|$)",
            txt,
            re.S,
        )
        if found:
            return found[-1].strip()
        return txt.strip()
