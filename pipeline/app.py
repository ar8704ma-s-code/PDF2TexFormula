import streamlit as st
from pathlib import Path
import zipfile
import tempfile
import threading
import torch
import json

from pipeline import CompletePipeline
from repair_head import RepairHead


# ============================================================
# Device choice
# ============================================================
def choose_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# Cached OCR pipeline loader (NO REPAIRHEAD INSIDE)
# ============================================================
@st.cache_resource
def load_pipeline(device: str):
    print(f"[Streamlit] Loading OCR pipeline on {device} ...")
    pipeline = CompletePipeline(device=device)
    print("[Streamlit] OCR pipeline ready.")
    return pipeline


# ============================================================
# Cached RepairHead (loaded ON MAIN THREAD)
# ============================================================
@st.cache_resource
def load_repair_head(device: str):
    print(f"[Streamlit] Loading RepairHead (Qwen) on {device} ...")
    repair = RepairHead(device=device, mode="local_1_5B")
    print("[Streamlit] RepairHead ready.")
    return repair


# ============================================================
# Background worker: ONLY runs process_pdf, NOT loading models
# ============================================================
def process_pdf_background(pipeline, pdf_path: Path, worker_state: dict):
    try:
        worker_state["status"] = "running_pipeline"
        pipeline.process_pdf(pdf_path)
        results_dir = Path("output_1.5B") / pdf_path.stem
        worker_state["results_dir"] = results_dir
        worker_state["status"] = "done"
    except Exception as e:
        worker_state["error"] = str(e)
        worker_state["status"] = "error"


# ============================================================
# Helpers for UI
# ============================================================
def render_pdf_first_page(pdf_path: Path):
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            return None
        pix = doc.load_page(0).get_pixmap(matrix=fitz.Matrix(2, 2))
        return pix.tobytes("png")
    except Exception as e:
        st.warning(f"PDF preview error: {e}")
        return None


def load_formulas_from_outputs(output_dir: Path):
    final_json = output_dir / "final_formulas.json"
    if final_json.exists():
        data = json.loads(final_json.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "formulas" in data:
            return data["formulas"]
        if isinstance(data, list):
            return data

    overall_json = output_dir / "overall_ocr_results.json"
    formulas = []
    if overall_json.exists():
        data = json.loads(overall_json.read_text(encoding="utf-8"))
        for _, page_info in data.items():
            formulas.extend(page_info.get("formulas", []))
    return formulas


def extract_latex_strings(formulas):
    latex_list = []
    for f in formulas:
        if isinstance(f, str):
            latex_list.append(f)
        elif isinstance(f, dict):
            for key in ("repaired", "raw_latex", "latex"):
                if key in f and f[key]:
                    latex_list.append(f[key])
                    break
            else:
                latex_list.append(str(f))
        else:
            latex_list.append(str(f))
    return latex_list


# ============================================================
# Streamlit App Layout
# ============================================================
st.set_page_config(page_title="LaTeX Repair Pipeline", layout="wide")
st.title("📘 PDF → Formula OCR & LaTeX Repair")

# Initialize state
if "status" not in st.session_state:
    st.session_state["status"] = "idle"
    st.session_state["work_dir"] = Path(tempfile.mkdtemp())
    st.session_state["pdf_path"] = None
    st.session_state["results_dir"] = None
    st.session_state["worker_state"] = None


# worker state syncing
worker_state = st.session_state.get("worker_state")
if worker_state:
    ws = worker_state.get("status")
    if ws == "done":
        st.session_state["status"] = "done"
        st.session_state["results_dir"] = worker_state["results_dir"]
    elif ws == "error":
        st.session_state["status"] = "error"
        st.session_state["error"] = worker_state["error"]
    elif ws in ("running_pipeline",):
        st.session_state["status"] = ws

status = st.session_state["status"]
work_dir = st.session_state["work_dir"]

left, right = st.columns([1, 2])

# ============================================================
# LEFT: Upload & Run
# ============================================================
with left:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Select PDF", type=["pdf"])

    if uploaded_file:
        pdf_path = work_dir / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())
        st.session_state["pdf_path"] = pdf_path
        st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("🚀 Run Pipeline"):
            device = choose_device()

            st.session_state["status"] = "loading_model"
            with st.spinner("Loading models..."):
                pipeline = load_pipeline(device)
                repair = load_repair_head(device)

            # Inject RepairHead into pipeline (lazy load done!)
            pipeline.set_repair_head(repair)

            # Start PDF processing
            worker_state = {"status": "running_pipeline", "results_dir": None}
            st.session_state["worker_state"] = worker_state

            thread = threading.Thread(
                target=process_pdf_background,
                args=(pipeline, pdf_path, worker_state),
                daemon=True,
            )
            thread.start()


# ============================================================
# RIGHT: Results
# ============================================================
with right:
    if status == "idle":
        st.info("Upload a PDF to begin.")

    elif status == "loading_model":
        st.info("⚙️ Loading OCR + Repair models...")

    elif status == "running_pipeline":
        st.info("🔍 Processing PDF (OCR + LaTeX Repair)...")

    elif status == "done":
        st.success("🎉 Pipeline completed!")

        results_dir = st.session_state["results_dir"]
        pdf_path = st.session_state["pdf_path"]

        # Preview PDF
        st.header("📄 PDF Preview")
        preview = render_pdf_first_page(pdf_path)
        if preview:
            st.image(preview)
        else:
            st.warning("Failed to render first page.")

        # Show formulas
        st.header("🧮 Extracted & Repaired Formulas")
        formulas = extract_latex_strings(load_formulas_from_outputs(results_dir))
        if not formulas:
            st.warning("No formulas found.")
        else:
            for i, tex in enumerate(formulas, 1):
                st.markdown(f"#### Formula {i}")
                try:
                    st.latex(tex)
                except:
                    st.code(tex)
                st.code(tex, language="latex")

        # Download ZIP
        st.header("📦 Download Results")
        zip_path = work_dir / f"{pdf_path.stem}_results.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in results_dir.rglob("*"):
                zf.write(f, f.relative_to(results_dir))

        st.download_button(
            "⬇ Download ZIP",
            zip_path.read_bytes(),
            file_name=f"{pdf_path.stem}_results.zip",
            mime="application/zip",
        )

        st.header("📊 Evaluation (coming soon)")
        st.info("Evaluation metrics will be added later.")

    elif status == "error":
        st.error(f"❌ Error: {st.session_state.get('error')}")
