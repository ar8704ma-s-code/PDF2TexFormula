import json
import re
from collections import defaultdict

# --------------------------
# Normalization
# --------------------------

def normalize_latex(s: str) -> str:
    """Normalize LaTeX to reduce superficial formatting differences."""
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\\begin\{.*?\}", "", s)
    s = re.sub(r"\\end\{.*?\}", "", s)
    s = s.replace("\\[", "").replace("\\]", "").replace("$$", "")
    for cmd in ["\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "\\thinspace", "\\ "]:
        s = s.replace(cmd, "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("{", "").replace("}", "")
    return s

# --------------------------
# LCS (for char-level and token-level)
# --------------------------

def lcs_len(a, b) -> int:
    """Compute LCS length for sequences (strings or token lists)."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0

    # Use DP with O(min(n,m)) memory
    if m > n:
        a, b = b, a
        n, m = m, n

    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[m]

def lcs_sim_char(a: str, b: str) -> float:
    """Character-level LCS similarity normalized by max length."""
    a = normalize_latex(a)
    b = normalize_latex(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    L = lcs_len(a, b)
    return L / max(len(a), len(b))

# --------------------------
# Levenshtein similarity (strict / edit-distance)
# --------------------------

def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Ensure a is shorter
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        cur = [i]
        for j, ca in enumerate(a, 1):
            cur.append(min(
                cur[j - 1] + 1,           # insert
                prev[j] + 1,              # delete
                prev[j - 1] + (ca != cb)  # substitute
            ))
        prev = cur
    return prev[-1]

def lev_sim(a: str, b: str) -> float:
    """Normalized Levenshtein similarity = 1 - edit_dist / max_len."""
    a = normalize_latex(a)
    b = normalize_latex(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = levenshtein(a, b)
    return 1.0 - dist / max(len(a), len(b))

# --------------------------
# Structural similarity: LCS over LaTeX command sequences
# --------------------------

_cmd_pat = re.compile(r"\\[a-zA-Z]+")  # commands like \hat, \max, \mathcal

def extract_commands(s: str):
    s = normalize_latex(s)
    return _cmd_pat.findall(s)

def structural_sim(a: str, b: str) -> float:
    """LCS similarity over LaTeX command lists, normalized by max list length."""
    A = extract_commands(a)
    B = extract_commands(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    L = lcs_len(A, B)
    return L / max(len(A), len(B))

# --------------------------
# Semantic token similarity: heuristic token overlap
# --------------------------

# Semantic tokens include LaTeX commands, identifiers, numbers, and restricted operators.
_sem_pat = re.compile(r"""
    \\[a-zA-Z]+            |  # LaTeX commands
    [a-zA-Z]+              |  # alphabetic identifiers
    \d+(?:\.\d+)?          |  # numeric literals
    (<=|>=|!=|==)          |  # multi-char operators
    [+\-*/=()_\^<>]           # restricted operators / structural symbols
""", re.VERBOSE)

def extract_semantic_tokens(s: str):
    s = normalize_latex(s)
    return _sem_pat.findall(s)

def semantic_token_sim(a: str, b: str) -> float:
    """
    Heuristic semantic similarity based on token overlap.
    Here: LCS similarity over semantic token sequences (order-aware), normalized by max length.
    """
    A = extract_semantic_tokens(a)
    B = extract_semantic_tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    L = lcs_len(A, B)
    return L / max(len(A), len(B))

# --------------------------
# Overall score (paper weights)
# --------------------------

W_SEM = 0.4
W_STRICT = 0.3   # Levenshtein
W_STRUCT = 0.2   # command LCS
W_ELEM = 0.1     # char LCS

def overall_score(sem, strict, struct, elem) -> float:
    return (W_SEM * sem) + (W_STRICT * strict) + (W_STRUCT * struct) + (W_ELEM * elem)

# --------------------------
# Main evaluation
# --------------------------

models = ["pix2tex", "1.5B", "7B"]

with open("test_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

stats = {m: defaultdict(list) for m in models}

for row in data:
    gt = row["gt"]
    for m in models:
        pred = row.get(m, "")
        sem = semantic_token_sim(pred, gt)
        strict = lev_sim(pred, gt)
        struct = structural_sim(pred, gt)
        elem = lcs_sim_char(pred, gt)
        comb = overall_score(sem, strict, struct, elem)

        stats[m]["sem"].append(sem)
        stats[m]["strict"].append(strict)
        stats[m]["struct"].append(struct)
        stats[m]["elem"].append(elem)
        stats[m]["comb"].append(comb)

def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

print("\n=== Repair Head Benchmark (GT only, weighted) ===")
for m in models:
    sem = avg(stats[m]["sem"])
    strict = avg(stats[m]["strict"])
    struct = avg(stats[m]["struct"])
    elem = avg(stats[m]["elem"])
    comb = avg(stats[m]["comb"])

    print(f"\n{m}")
    print(f"  Semantic Token Similarity : {sem:.4f}")
    print(f"  Edit-distance (Levenshtein): {strict:.4f}")
    print(f"  Structural (cmd LCS)      : {struct:.4f}")
    print(f"  Character LCS             : {elem:.4f}")
    print(f"  Score_comb (0.4/0.3/0.2/0.1): {comb:.4f}")
