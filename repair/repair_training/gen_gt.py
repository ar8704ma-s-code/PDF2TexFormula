import tarfile
import json
import re
from pathlib import Path
from tqdm import tqdm

# 原始 tex 的 tar.gz
TEX_TGZ_DIR = Path("data/arxiv_math_test/tex")

# 第一步输出：raw gt
GT_DIR = Path("data/arxiv_math_test/gt")
GT_DIR.mkdir(parents=True, exist_ok=True)

# 第二步输出：严格清洗后的 gt (新文件夹，只包含新处理的)
GT_CLEAN_DIR_NEW = Path("data/arxiv_math_test/gt_cleaned_2")
GT_CLEAN_DIR_NEW.mkdir(parents=True, exist_ok=True)

# 原有的 cleaned gt 目录
GT_CLEAN_DIR_OLD = Path("data/arxiv_math_test/gt_cleaned")

# ============================================================
#  Part 1: 从 tar.gz 提取公式
# ============================================================

def clean_paper_id(tgz: Path) -> str:
    """去掉 .tar.gz，确保 paper_id 是正确的 arXiv ID"""
    name = tgz.name
    return re.sub(r"\.tar\.gz$", "", name)


def read_all_tex_from_tar(tar_path: Path) -> str:
    contents = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lower().endswith(".tex"):
                f = tar.extractfile(m)
                if f:
                    contents.append(f.read().decode("utf-8", errors="ignore"))
    return "\n".join(contents)


def extract_equations_from_tex(tex: str) -> list[str]:
    eqs = []

    envs = [
        "equation", "equation*", "align", "align*",
        "gather", "gather*", "multline", "multline*",
        "eqnarray", "eqnarray*", "aligned", "aligned*",
        "alignat", "alignat*"
    ]

    env_pattern = re.compile(
        r"\\begin\{(" + "|".join(envs) + r")\}(.*?)\\end\{\1\}",
        re.DOTALL
    )
    for m in env_pattern.finditer(tex):
        eqs.append(m.group(0).strip())

    for m in re.finditer(r"\\\[(.*?)\\\]", tex, re.DOTALL):
        eqs.append(m.group(0).strip())

    for m in re.finditer(r"\$\$(.*?)\$\$", tex, re.DOTALL):
        eqs.append(m.group(0).strip())

    return eqs


def build_gt_for_paper(tgz: Path):
    pid = clean_paper_id(tgz)

    tex = read_all_tex_from_tar(tgz)
    equations = extract_equations_from_tex(tex)

    out = GT_DIR / f"{pid}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, eq in enumerate(equations):
            f.write(json.dumps({
                "paper_id": pid,
                "gt_id": i,
                "latex_gt": eq
            }, ensure_ascii=False) + "\n")

    return len(equations)

# ============================================================
# Part 2: 严格清洗 — 删除所有垃圾公式
# ============================================================

ENV_PREFIX_ONLY = re.compile(r"^\s*\\begin\{[a-zA-Z*]+\}\s*$")
ENV_WITH_EXTRA_BRACE = re.compile(r"^\s*\\begin\{[a-zA-Z*]+\}\}?\s*$")
BRACKET_EMPTY = re.compile(r"^\s*\\\[\s*\\?\]\s*$")
DOLLAR_EMPTY = re.compile(r"^\s*\$\$\s*\$\$\s*$")


def has_math_content(latex: str) -> bool:
    body = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", latex, flags=re.DOTALL).strip()

    if len(body) < 2:
        return False

    indicators = [
        r'[a-zA-Z]\s*[=+\-*/]',
        r'\\frac',
        r'\\sum',
        r'\\int',
        r'[0-9]',
        r'\\alpha|\\beta|\\gamma',
        r'[_^]\{.*?\}',
    ]
    return any(re.search(p, body) for p in indicators)


def should_remove(latex: str) -> bool:
    s = latex.strip()

    if ENV_PREFIX_ONLY.match(s):
        return True
    if ENV_WITH_EXTRA_BRACE.match(s):
        return True
    if BRACKET_EMPTY.match(s):
        return True
    if DOLLAR_EMPTY.match(s):
        return True
    if not has_math_content(s):
        return True

    return False


def clean_gt_file(in_path: Path, out_path: Path):
    keep = []
    removed = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            latex = obj["latex_gt"]

            if should_remove(latex):
                removed += 1
            else:
                keep.append(obj)

    for i, o in enumerate(keep):
        o["gt_id"] = i

    out_path.write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in keep),
        encoding="utf-8"
    )

    return len(keep), removed


def get_processed_paper_ids():
    """获取已经处理过的论文ID列表"""
    processed_ids = set()
    
    # 从原有的 gt_cleaned 目录获取已处理的论文ID
    if GT_CLEAN_DIR_OLD.exists():
        for jsonl_file in GT_CLEAN_DIR_OLD.glob("*.jsonl"):
            paper_id = jsonl_file.stem
            processed_ids.add(paper_id)
    
    print(f"Found {len(processed_ids)} already processed papers in gt_cleaned/")
    return processed_ids


# ============================================================
# Main
# ============================================================

def main():
    # 获取已经处理过的论文ID
    processed_ids = get_processed_paper_ids()
    
    print("Extracting raw GT for new papers only...")
    tgz_files = sorted(TEX_TGZ_DIR.glob("*.gz"))
    
    new_papers_count = 0
    skipped_papers_count = 0
    
    # 第一步：只提取新论文的原始GT
    for tgz in tqdm(tgz_files, desc="Extracting raw GT"):
        paper_id = clean_paper_id(tgz)
        
        # 跳过已经处理过的论文
        if paper_id in processed_ids:
            skipped_papers_count += 1
            continue
        
        # 处理新论文
        try:
            eq_count = build_gt_for_paper(tgz)
            new_papers_count += 1
            print(f"  {paper_id}: extracted {eq_count} equations")
        except Exception as e:
            print(f"  Error processing {paper_id}: {e}")
            continue

    print(f"\nExtraction complete: {new_papers_count} new papers, {skipped_papers_count} skipped")

    # 第二步：只清洗新提取的GT文件
    print("Cleaning new GT files...")
    new_gt_files = []
    for gt_file in GT_DIR.glob("*.jsonl"):
        paper_id = gt_file.stem
        # 只处理不在已处理列表中的文件（即新提取的）
        if paper_id not in processed_ids:
            new_gt_files.append(gt_file)
    
    if not new_gt_files:
        print("No new GT files to clean.")
        return
    
    total_kept = 0
    total_removed = 0
    
    for gt in tqdm(new_gt_files, desc="Cleaning GT"):
        out = GT_CLEAN_DIR_NEW / gt.name
        keep, rm = clean_gt_file(gt, out)
        total_kept += keep
        total_removed += rm
        print(f"  {gt.name}: kept {keep}, removed {rm}")

    print(f"\nFinal results:")
    print(f"  New papers processed: {len(new_gt_files)}")
    print(f"  Total equations kept: {total_kept}")
    print(f"  Total equations removed: {total_removed}")
    print(f"  New cleaned GT saved to: {GT_CLEAN_DIR_NEW}")


if __name__ == "__main__":
    main()