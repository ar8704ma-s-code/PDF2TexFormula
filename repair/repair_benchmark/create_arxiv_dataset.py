import json
import time
import requests
from pathlib import Path
import xml.etree.ElementTree as ET
import re

SAVE_DIR = Path("data/arxiv_math_test")
(SAVE_DIR / "pdf").mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "tex").mkdir(exist_ok=True)

N = 10   # number of papers you want
CATEGORY = ["math","physics"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/x-eprint-tar,*/*"
}

# 新增：ID管理文件
PAPER_IDS_FILE = SAVE_DIR / "downloaded_paper_ids.txt"
PAPER_IDS_FILE.touch(exist_ok=True)  # 确保文件存在


# ============================================================
# NEW: 加载 data/arxiv_math/pdf 下已有 ID
# ============================================================
def load_local_pdf_ids():
    """读取 data/arxiv_math/pdf 中已有 PDF 并提取 arXiv ID"""
    pdf_dir = Path("data/arxiv_math/pdf")
    ids = set()

    if not pdf_dir.exists():
        print("No local pdf directory found.")
        return ids

    for f in pdf_dir.glob("*.pdf"):
        name = f.stem  # remove .pdf

        # 新格式 2511.21644v1
        m = re.match(r"(\d{4}\.\d{4,5}v\d+)$", name)
        if m:
            ids.add(m.group(1))
            continue

        # 老格式 math_0501234v1 → math/0501234v1
        m = re.match(r"([a-zA-Z]+)_(\d{7}v\d+)$", name)
        if m:
            ids.add(f"{m.group(1)}/{m.group(2)}")
            continue

        ids.add(name)

    print(f"Loaded {len(ids)} local PDF IDs from data/arxiv_math/pdf")
    return ids


# ============================================================
# 原有函数（保持不变）
# ============================================================
def load_existing_ids():
    if PAPER_IDS_FILE.exists():
        with PAPER_IDS_FILE.open('r', encoding='utf-8') as f:
            ids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(ids)} existing paper IDs")
        return ids
    return set()


def save_paper_id(paper_id):
    with PAPER_IDS_FILE.open('a', encoding='utf-8') as f:
        f.write(paper_id + '\n')


def check_local_files_exist(paper_id):
    filename = paper_id.replace('/', '_')
    pdf_path = SAVE_DIR / "pdf" / f"{filename}.pdf"
    tex_path = SAVE_DIR / "tex" / f"{filename}.tar.gz"

    pdf_exists = pdf_path.exists() and pdf_path.stat().st_size > 50000
    tex_exists = tex_path.exists() and tex_path.stat().st_size > 10000

    return pdf_exists and tex_exists


def search_arxiv_recent(category="math", limit=100):
    print(f"Searching arXiv for recent {category} papers...")
    
    base_url = "https://export.arxiv.org/api/query"

    params = {
        'search_query': f'cat:{category}',
        'start': 0,
        'max_results': limit,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    try:
        response = requests.get(base_url, params=params, headers=HEADERS, timeout=30)
        print(f"Response status: {response.status_code}")

        if response.status_code != 200:
            return []

        content = response.text
        if len(content) < 100:
            return []

        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        print(f"Found {len(entries)} entries")
        
        papers = []
        for entry in entries:
            id_elem = entry.find('atom:id', ns)
            if id_elem is None:
                continue

            arxiv_url = id_elem.text
            match = re.search(r'arxiv\.org/abs/([\d\.v]+)', arxiv_url)
            if match:
                paper_id = match.group(1)

                title_elem = entry.find('atom:title', ns)
                summary_elem = entry.find('atom:summary', ns)
                published_elem = entry.find('atom:published', ns)

                papers.append({
                    'id': paper_id,
                    'title': title_elem.text.strip() if title_elem else "No title",
                    'published': published_elem.text if published_elem else "No date",
                    'abstract': (summary_elem.text[:200] + "...") if summary_elem else "No abstract"
                })
        
        return papers

    except:
        return []


def download_pdf(paper_info, out_dir):
    pid = paper_info['id']

    urls = [
        f"https://arxiv.org/pdf/{pid}.pdf",
        f"https://arxiv.org/pdf/{pid}",
    ]

    for url in urls:
        try:
            print(f"   PDF: {url}")
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 200:
                content = response.content
                if len(content) > 50000 and content[:4] == b'%PDF':
                    filename = f"{pid.replace('/', '_')}.pdf"
                    (out_dir / filename).write_bytes(content)
                    print(f"   PDF OK: {len(content):,} bytes")
                    return True
        except:
            pass
    
    return False


def download_tex(paper_info, out_dir):
    pid = paper_info['id']

    urls = [
        f"https://arxiv.org/e-print/{pid}",
        f"https://arxiv.org/src/{pid}",
    ]

    for url in urls:
        try:
            print(f"   TeX: {url}")
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 200 and len(response.content) > 10000:
                filename = f"{pid.replace('/', '_')}.tar.gz"
                (out_dir / filename).write_bytes(response.content)
                print(f"   TeX OK: {len(response.content):,} bytes")
                return True
        except:
            pass

    return False


def check_paper_exists(paper_id):
    url = f"https://arxiv.org/abs/{paper_id}"
    try:
        response = requests.head(url, headers=HEADERS, timeout=10)
        return response.status_code == 200
    except:
        return False


# ============================================================
# MAIN —— 整合完整的“不重复下载”逻辑
# ============================================================
def main():
    print("Starting arXiv dataset creation...")

    existing_ids = load_existing_ids()
    local_pdf_ids = load_local_pdf_ids()  # NEW

    print(f"Found {len(existing_ids)} IDs in downloaded_paper_ids.txt")
    print(f"Found {len(local_pdf_ids)} IDs in data/arxiv_math/pdf")

    papers = search_arxiv_recent(CATEGORY, limit=100)

    usable = []
    skipped = []
    duplicate_count = 0

    for i, paper in enumerate(papers):
        if len(usable) >= N:
            break

        pid = paper['id']
        print(f"\n{'='*50}")
        print(f"[{i+1}/{len(papers)}] {pid}")

        # ============================================================
        # NEW!!!!!  跳过已有 PDF 的论文
        # ============================================================
        if pid in local_pdf_ids:
            print("   ⏩ SKIP: Already in data/arxiv_math/pdf")
            duplicate_count += 1
            continue

        if pid in existing_ids:
            print("   ⏩ SKIP: Already in downloaded records")
            duplicate_count += 1
            continue

        if check_local_files_exist(pid):
            print("   ⏩ SKIP: Local files already exist (pdf+tex)")
            save_paper_id(pid)
            existing_ids.add(pid)
            duplicate_count += 1
            continue

        if not check_paper_exists(pid):
            print("   ❌ Paper not found on arXiv")
            skipped.append(pid)
            continue

        # 下载文件
        success_pdf = download_pdf(paper, SAVE_DIR / "pdf")
        time.sleep(1)
        success_tex = download_tex(paper, SAVE_DIR / "tex")
        time.sleep(1)

        if success_pdf and success_tex:
            usable.append(paper)
            save_paper_id(pid)
            existing_ids.add(pid)
            print("   ✅ SUCCESS")
        else:
            skipped.append(pid)
            print("   ❌ SKIPPED (missing files)")

    print("\nDownload finished.")
    print(f"New usable papers: {len(usable)}")
    print(f"Duplicate: {duplicate_count}, Failed: {len(skipped)}")


if __name__ == "__main__":
    main()
