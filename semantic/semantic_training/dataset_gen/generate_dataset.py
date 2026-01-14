import json
import random
import os
import sys

# Load local OCR noise tools
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(CURRENT_DIR)

from ocr_noise import apply_ocr_noise, corrupt_formula
from arxiv_extractor import generate_arxiv_dataset


###############################################################
# ============  PART 1: Simple TEMPLATE datasets  ============
###############################################################
CHEM_TEMPLATES = [
    "H2 + O2 -> H2O",
    "C + O2 -> CO2",
    "N2 + 3H2 -> 2NH3",
    "2H2O2 -> 2H2O + O2",
]

def generate_chem_samples(n=5000):
    for _ in range(n):
        base = random.choice(CHEM_TEMPLATES)
        if random.random() < 0.5:
            yield {"formula": base, "correct": True}
        else:
            yield {"formula": apply_ocr_noise(base), "correct": False}


MATH_TEMPLATES = [
    r"E = mc^2",
    r"F = ma",
    r"\frac{d}{dx} x^2 = 2x",
    r"a^2 + b^2 = c^2",
    r"\nabla \cdot E = \frac{\rho}{\epsilon_0}",
]

def generate_math_samples(n=5000):
    for _ in range(n):
        base = random.choice(MATH_TEMPLATES)
        if random.random() < 0.5:
            yield {"formula": base, "correct": True}
        else:
            yield {"formula": apply_ocr_noise(base), "correct": False}


MED_TEMPLATES = [
    "BMI = weight / height^2",
    "MAP = (SBP + 2*DBP) / 3",
    "CrCl = ((140 - age) * weight) / (72 * Scr)",
    "pH = pKa + log([A-]/[HA])",
]

def generate_med_samples(n=5000):
    for _ in range(n):
        base = random.choice(MED_TEMPLATES)
        if random.random() < 0.5:
            yield {"formula": base, "correct": True}
        else:
            yield {"formula": apply_ocr_noise(base), "correct": False}


PHYSICS_TEMPLATES = [
    r"F = G \frac{m_1 m_2}{r^2}",
    r"\lambda = \frac{h}{p}",
    r"E = h\nu",
    r"V = IR",
]

def generate_physics_samples(n=5000):
    for _ in range(n):
        base = random.choice(PHYSICS_TEMPLATES)
        if random.random() < 0.5:
            yield {"formula": base, "correct": True}
        else:
            yield {"formula": apply_ocr_noise(base), "correct": False}


###############################################################
# ===============  PART 2: Paths + Utilities  ================
###############################################################
BASE_DIR = "/home/baiyinyou/workspace/SmolLatexFormula/src/training/data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def save_jsonl(path, generator):
    """Write generator output into JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in generator:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Saved] {path}")


def split_and_save_dataset(generator, total_samples, base_filename):
    """
    Split dataset into train/validation/test and save to respective directories
    Ratio: 80% train, 10% validation, 10% test
    """
    # Collect all samples
    all_samples = list(generator)
    
    # Shuffle the dataset
    random.shuffle(all_samples)
    
    # Calculate split indices
    train_end = int(total_samples * 0.8)
    val_end = train_end + int(total_samples * 0.1)
    
    # Split the data
    train_samples = all_samples[:train_end]
    val_samples = all_samples[train_end:val_end]
    test_samples = all_samples[val_end:total_samples]
    
    # Save to respective directories
    save_jsonl(os.path.join(TRAIN_DIR, f"{base_filename}_train.jsonl"), train_samples)
    save_jsonl(os.path.join(VAL_DIR, f"{base_filename}_val.jsonl"), val_samples)
    save_jsonl(os.path.join(TEST_DIR, f"{base_filename}_test.jsonl"), test_samples)
    
    print(f"[Split] {base_filename}: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    return len(train_samples), len(val_samples), len(test_samples)


def merge_jsonl_files(files, out_path):
    """Merge multiple JSONL files."""
    with open(out_path, "w", encoding="utf-8") as fout:
        for file in files:
            with open(file, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"[Merged] → {out_path}")


###############################################################
# ============  PART 3: arXiv REAL FORMULAS  ============
###############################################################
def generate_arxiv_samples(fields=None, papers_per_field=50):
    """
    fields: list of arXiv subjects
    papers_per_field: how many papers to scrape per subject
    """
    if fields is None:
        fields = ["physics", "math", "chemistry", "medical"]

    print(f"[arXiv] Scraping fields: {fields}")

    for formula in generate_arxiv_dataset(fields=fields, papers_per_field=papers_per_field):
        if random.random() < 0.5:
            yield {"formula": formula, "correct": True}
        else:
            yield {"formula": corrupt_formula(formula), "correct": False}


###############################################################
# ============  FINAL PIPELINE  (MAIN ENTRY)  ================
###############################################################
def generate_all_datasets():
    print("\n===== STEP 1: Generating and splitting template datasets =====")
    
    # Generate and split each dataset type
    math_train, math_val, math_test = split_and_save_dataset(
        generate_math_samples(5000), 5000, "math"
    )
    
    chem_train, chem_val, chem_test = split_and_save_dataset(
        generate_chem_samples(5000), 5000, "chem"
    )
    
    med_train, med_val, med_test = split_and_save_dataset(
        generate_med_samples(5000), 5000, "med"
    )
    
    physics_train, physics_val, physics_test = split_and_save_dataset(
        generate_physics_samples(5000), 5000, "physics"
    )

    print("\n===== STEP 2: Generating and splitting arXiv dataset =====")
    arxiv_samples = list(generate_arxiv_samples(fields=None, papers_per_field=100))
    arxiv_train, arxiv_val, arxiv_test = split_and_save_dataset(
        arxiv_samples, len(arxiv_samples), "arxiv"
    )

    print("\n===== STEP 3: Merging ALL datasets by split =====")
    
    # Merge all train files
    train_files = [
        os.path.join(TRAIN_DIR, "math_train.jsonl"),
        os.path.join(TRAIN_DIR, "chem_train.jsonl"),
        os.path.join(TRAIN_DIR, "med_train.jsonl"),
        os.path.join(TRAIN_DIR, "physics_train.jsonl"),
        os.path.join(TRAIN_DIR, "arxiv_train.jsonl")
    ]
    merge_jsonl_files(train_files, os.path.join(TRAIN_DIR, "merged_train.jsonl"))
    
    # Merge all validation files
    val_files = [
        os.path.join(VAL_DIR, "math_val.jsonl"),
        os.path.join(VAL_DIR, "chem_val.jsonl"),
        os.path.join(VAL_DIR, "med_val.jsonl"),
        os.path.join(VAL_DIR, "physics_val.jsonl"),
        os.path.join(VAL_DIR, "arxiv_val.jsonl")
    ]
    merge_jsonl_files(val_files, os.path.join(VAL_DIR, "merged_val.jsonl"))
    
    # Merge all test files
    test_files = [
        os.path.join(TEST_DIR, "math_test.jsonl"),
        os.path.join(TEST_DIR, "chem_test.jsonl"),
        os.path.join(TEST_DIR, "med_test.jsonl"),
        os.path.join(TEST_DIR, "physics_test.jsonl"),
        os.path.join(TEST_DIR, "arxiv_test.jsonl")
    ]
    merge_jsonl_files(test_files, os.path.join(TEST_DIR, "merged_test.jsonl"))

    print("\n===== STEP 4: Creating final merged dataset (optional) =====")
    # Create one final merged dataset for backward compatibility
    final_merged_path = os.path.join(BASE_DIR, "merged_total.jsonl")
    merge_jsonl_files([
        os.path.join(TRAIN_DIR, "merged_train.jsonl"),
        os.path.join(VAL_DIR, "merged_val.jsonl"),
        os.path.join(TEST_DIR, "merged_test.jsonl")
    ], final_merged_path)

    # Print summary
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    print(f"Training set:    {math_train + chem_train + med_train + physics_train + arxiv_train} samples")
    print(f"Validation set:  {math_val + chem_val + med_val + physics_val + arxiv_val} samples")
    print(f"Test set:        {math_test + chem_test + med_test + physics_test + arxiv_test} samples")
    print(f"Total:           {math_train+math_val+math_test + chem_train+chem_val+chem_test + med_train+med_val+med_test + physics_train+physics_val+physics_test + arxiv_train+arxiv_val+arxiv_test} samples")
    print("\nFile locations:")
    print(f"Train:     {TRAIN_DIR}/merged_train.jsonl")
    print(f"Validation: {VAL_DIR}/merged_val.jsonl")
    print(f"Test:      {TEST_DIR}/merged_test.jsonl")
    print(f"Combined:  {final_merged_path}")
    print("="*50)


if __name__ == "__main__":
    generate_all_datasets()