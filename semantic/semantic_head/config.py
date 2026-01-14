# src/semantic_head/config.py
from dataclasses import dataclass
from typing import List

@dataclass
class SemanticHeadConfig:
    # Path to your LoRA-tuned model
    model_path: str = "/home/baiyinyou/workspace/SmolLatexFormula/models/Qwen_formula_lora"
    base_model: str = "Qwen/Qwen1.5-0.5B"

    max_length: int = 128       # max length for prompt
    max_new_tokens: int = 4     # we only need "correct"/"incorrect"
    temperature: float = 0.0    
    top_p: float = 1.0
    device: str = "auto"

   
    error_types: List[str] = None

    def __post_init__(self):
        if self.error_types is None:
            self.error_types = [
                "syntax_error",
                "parameter_error",
                "semantic_error",
                "notation_error",
                "logic_error",
                "dimensional_error",
            ]
