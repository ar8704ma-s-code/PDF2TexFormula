import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from .config import SemanticHeadConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormulaSemanticHead:
    """
    Evaluates formulas by generating only one token:
    'correct' or 'incorrect'
    Matching your trained LoRA.
    """

    def __init__(self, config: SemanticHeadConfig):
        self.config = config
        self.device = self._setup_device()
        self.model, self.tokenizer = self._load_model()
        self.prompt_template = self._create_prompt()

    def _setup_device(self):
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _load_model(self):
        logger.info(f"Loading base model: {self.config.base_model}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info(f"Loading LoRA weights: {self.config.model_path}")
        model = PeftModel.from_pretrained(
            base,
            self.config.model_path,
            device_map="auto"
        )
        model.eval()

        return model, tokenizer

    def _create_prompt(self):
        """
        EXACTLY the same structure as training data.
        """
        return (
            "Verify the correctness of the formula.\n"
            "Answer only with 'correct' or 'incorrect'.\n\n"
            "Formula: {formula}\n"
            "Correctness:"
        )

    def analyze_formula(self, formula: str) -> dict:
        """
        Generate exactly one classification word.
        """

        prompt = self.prompt_template.format(formula=formula)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = out[0][input_len:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

        if "incorrect" in result:
            label = False
        elif "correct" in result:
            label = True
        else:
            label = None

        return {
            "formula": formula,
            "is_correct": label,
            "raw_output": result,
            "confidence": 0.9 if label is not None else 0.3
        }
