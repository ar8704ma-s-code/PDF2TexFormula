import requests
import tarfile
import io
from bs4 import BeautifulSoup
import re
import random

############################################################
# 1. HTML SCRAPER：抓取某个领域最新论文 ID
############################################################

def scrape_arxiv_ids(category="physics", max_results=20):
    """
    Scrape arXiv newest submissions HTML (not API).
    Always returns IDs (unless totally offline).
    """
    url = f"https://arxiv.org/list/{category}/new"
    print(f"[HTML] Fetching: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SmolLatexFormula/1.0)"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            print(f"[ERROR] Could not load HTML, status={resp.status_code}")
            return []
    except Exception as e:
        print(f"[ERROR] HTML fetch failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    dt_list = soup.find_all("dt")
    ids = []

    for dt in dt_list:
        a = dt.find("a", title="Abstract")
        if a:
            arxid = a["href"].split("/")[-1]
            ids.append(arxid)
        if len(ids) >= max_results:
            break

    print(f"[INFO] Scraped {len(ids)} IDs for field: {category}")
    return ids


############################################################
# 2. 下载 arXiv tar 源码包
############################################################

def fetch_arxiv_tex(arxiv_id):
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    print(f"[Download] {arxiv_id}")

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"[Failed] {arxiv_id} (status={resp.status_code})")
            return None
        return io.BytesIO(resp.content)
    except Exception as e:
        print(f"[Error] downloading {arxiv_id}: {e}")
        return None


############################################################
# 3. 解压 tar 读取 .tex 文件
############################################################

def extract_tex_files(raw_tar):
    tex_files = []
    try:
        with tarfile.open(fileobj=raw_tar, mode="r:*") as tar:
            for m in tar.getmembers():
                if m.name.endswith(".tex"):
                    try:
                        content = tar.extractfile(m).read().decode("utf-8", errors="ignore")
                        tex_files.append(content)
                    except:
                        continue
    except:
        return []

    return tex_files


############################################################
# 4. 提取 TeX 中的公式
############################################################

def extract_latex_formulas(tex_str):
    formulas = []

    formulas.extend(re.findall(r"\$(.+?)\$", tex_str))                   # inline
    formulas.extend(re.findall(r"\\\[(.+?)\\\]", tex_str, re.DOTALL))    # \[ \]
    formulas.extend(re.findall(
        r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}",
        tex_str, re.DOTALL))                                             # equation env

    cleaned = []
    for f in formulas:
        f = f.strip()
        if 3 < len(f) < 400:
            cleaned.append(f)

    return cleaned


############################################################
# 5. ★ 多领域 arXiv 采集（你 generate_dataset.py 用到这个）
############################################################

def generate_arxiv_dataset(fields=None, papers_per_field=150):
    """
    fields: list of arXiv subjects
    papers_per_field: how many papers per subject to scrape
    """

    if fields is None:
        fields = ["physics", "math", "cancer", "chemistry", "medicine"]

    print(f"[arXiv] Multi-field scrape: {fields} (each {papers_per_field} papers)")

    for field in fields:
        ids = scrape_arxiv_ids(field, papers_per_field)
        print(f"[{field}] IDs: {ids}")

        for arxid in ids:
            raw = fetch_arxiv_tex(arxid)
            if not raw:
                continue

            tex_files = extract_tex_files(raw)
            for t in tex_files:
                for f in extract_latex_formulas(t):
                    yield f
