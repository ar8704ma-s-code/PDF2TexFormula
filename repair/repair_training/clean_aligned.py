"""
Filter Dataset - Keep Only Required Fields
This script filters the dataset to keep only:
task, raw_ocr, gt_latex, similarity, strict_similarity, semantic_similarity, structural_similarity, element_overlap
For repair_no_gt, pseudo_gt becomes gt_latex
"""

import json
from pathlib import Path
from tqdm import tqdm


# ==================================================
# Configuration
# ==================================================
REFINED_IN_DIR = Path("data/aligned_datasets/aligned_dataset_raw")  # Input directory
FILTERED_OUT_DIR = Path("data/aligned_datasets/aligned_dataset_filtered")  # Output directory
FILTERED_OUT_DIR.mkdir(parents=True, exist_ok=True)


def filter_record_fields(record: dict) -> dict:
    """
    Filter record to keep only required fields.
    
    Args:
        record: Original record
        
    Returns:
        Filtered record with only required fields
    """
    filtered = {}
    
    # Required fields
    filtered["task"] = record.get("task", "")
    filtered["raw_ocr"] = record.get("raw_ocr", "") or ""
    
    # Handle gt_latex field - use pseudo_gt for repair_no_gt tasks
    if filtered["task"] == "repair_no_gt":
        filtered["gt_latex"] = record.get("pseudo_gt", "") or ""
    else:
        filtered["gt_latex"] = record.get("gt_latex", "") or ""
    
    # Similarity scores
    filtered["similarity"] = record.get("similarity", -1.0)
    filtered["strict_similarity"] = record.get("strict_similarity", -1.0)
    filtered["semantic_similarity"] = record.get("semantic_similarity", -1.0)
    filtered["structural_similarity"] = record.get("structural_similarity", -1.0)
    filtered["element_overlap"] = record.get("element_overlap", -1.0)
    
    return filtered


def process_single_file(input_path: Path, output_path: Path) -> tuple:
    """
    Process a single JSONL file to filter fields.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        
    Returns:
        Tuple of (total_count, filtered_count, stats)
    """
    filtered_records = []
    total_count = 0
    stats = {
        "with_gt": 0,
        "repair_no_gt": 0,
        "noise": 0
    }
    
    try:
        with input_path.open('r', encoding='utf-8') as f_in:
            for line in f_in:
                total_count += 1
                record = json.loads(line)
                
                # Filter the record fields
                filtered_record = filter_record_fields(record)
                filtered_records.append(filtered_record)
                
                # Update statistics
                task_type = filtered_record["task"]
                if task_type in stats:
                    stats[task_type] += 1
        
        # Write filtered records to output file
        with output_path.open('w', encoding='utf-8') as f_out:
            for record in filtered_records:
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                
        return total_count, len(filtered_records), stats
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return total_count, 0, stats


def main():
    """
    Main function to process all files and filter fields.
    """
    # Get all input files
    input_files = sorted(REFINED_IN_DIR.glob("*.jsonl"))
    
    if not input_files:
        print(f"No JSONL files found in {REFINED_IN_DIR}")
        return
    
    # Global statistics
    global_stats = {
        "total_files": 0,
        "total_records": 0,
        "total_filtered": 0,
        "task_distribution": {
            "with_gt": 0,
            "repair_no_gt": 0, 
            "noise": 0
        }
    }
    
    # Process each file
    for input_file in tqdm(input_files, desc="Filtering files"):
        output_file = FILTERED_OUT_DIR / input_file.name
        
        total_count, filtered_count, file_stats = process_single_file(input_file, output_file)
        
        # Update global statistics
        global_stats["total_files"] += 1
        global_stats["total_records"] += total_count
        global_stats["total_filtered"] += filtered_count
        
        for task_type, count in file_stats.items():
            global_stats["task_distribution"][task_type] += count
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET FILTERING SUMMARY")
    print("="*50)
    print(f"Files processed: {global_stats['total_files']}")
    print(f"Total records: {global_stats['total_records']}")
    print(f"Filtered records: {global_stats['total_filtered']}")
    print(f"\nTask distribution:")
    for task_type, count in global_stats["task_distribution"].items():
        percentage = (count / global_stats['total_records']) * 100 if global_stats['total_records'] > 0 else 0
        print(f"  {task_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nFiltered data saved to: {FILTERED_OUT_DIR}")


# ==================================================
# Verification function
# ==================================================
def verify_filtered_format():
    """
    Verify that the filtered format is correct by sampling records.
    """
    filtered_files = list(FILTERED_OUT_DIR.glob("*.jsonl"))
    
    if not filtered_files:
        print("No filtered files found for verification")
        return
    
    sample_file = filtered_files[0]
    print(f"\nVerifying format for: {sample_file.name}")
    
    with sample_file.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Check first 3 records
                break
                
            record = json.loads(line)
            print(f"\nRecord {i + 1}:")
            print(f"  Task: {record.get('task')}")
            print(f"  Raw OCR: {record.get('raw_ocr')}")
            print(f"  GT LaTeX: {record.get('gt_latex')}")
            print(f"  Similarity: {record.get('similarity')}")
            print(f"  Strict Similarity: {record.get('strict_similarity')}")
            print(f"  Semantic Similarity: {record.get('semantic_similarity')}")
            print(f"  Structural Similarity: {record.get('structural_similarity')}")
            print(f"  Element Overlap: {record.get('element_overlap')}")
            
            # Check field count
            field_count = len(record.keys())
            expected_fields = 8
            if field_count == expected_fields:
                print(f"  ✅ Correct field count: {field_count}")
            else:
                print(f"  ⚠️  Wrong field count: {field_count} (expected: {expected_fields})")


if __name__ == "__main__":
    main()
    
    # Verify the results
    verify_filtered_format()