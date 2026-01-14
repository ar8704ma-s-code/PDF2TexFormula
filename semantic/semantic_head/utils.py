import json
from typing import List, Dict
import matplotlib.pyplot as plt
import pip as sns

def save_report(report: Dict, output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def plot_error_distribution(results: List[Dict], save_path: str = None):
    error_types = {}
    for result in results:
        if not result['is_correct']:
            error_type = result['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(error_types.keys()), y=list(error_types.values()))
        plt.title('Formula Error Type Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def filter_high_confidence_errors(results: List[Dict], threshold: float = 0.8) -> List[Dict]:
    return [r for r in results if not r['is_correct'] and r['confidence'] >= threshold]