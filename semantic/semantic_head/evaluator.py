# src/semantic_head/evaluator.py

import json
from typing import List
from tqdm.auto import tqdm

from .config import SemanticHeadConfig
from .model import FormulaSemanticHead


class FormulaEvaluator:
    """
    Evaluate a list of formulas using the FormulaSemanticHead.
    """

    def __init__(self, config: SemanticHeadConfig = None):
        self.config = config or SemanticHeadConfig()
        self.head = FormulaSemanticHead(self.config)

    def evaluate_formulas(self, formulas: List[str], output_path: str = None):
        results = []

        for f in tqdm(formulas, desc="Evaluating"):
            f = f.strip()
            if not f:
                continue

            result = self.head.analyze_formula(f)
            results.append(result)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results
