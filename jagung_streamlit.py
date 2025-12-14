import streamlit as st
import os
import io
import glob
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# PREPROCESS INPUT (sesuai backbone)
# =========================
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep

# =========================
# LANGCHAIN (RAG)
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# =========================
# KONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Sistem Pakar Jagung (Research Mode)", layout="wide")

if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# =========================
# PATH AMAN (local + cloud)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOP_FOLDER = os.path.join(BASE_DIR, "knowledge_base")

# =========================
# LABEL KELAS (URUTAN WAJIB)
# =========================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)",  # 0
    "Karat Daun (Common Rust)",           # 1
    "Bercak Daun (Gray Leaf Spot)",       # 2
    "Tanaman Sehat"                       # 3
]
N_CLASS = len(CLASS_NAMES)

# =========================
# MODEL FILES (.keras)
# =========================
MODEL_FILES = {
    "MobileNetV3": "model_jagung_mobilenetv3_vFinal.keras",
    "EfficientNet": "model_jagung_efficientnet_vFinal.keras",
    "DenseNet": "model_jagung_densenet_vFinal.keras",
}

PREPROCESS_MAP = {
    "MobileNetV3": mobilenet_prep,
    "EfficientNet": efficientnet_prep,
    "DenseNet": densenet_prep,
}

# =========================
# BOBOT AUTO (dari grafik F1 yang kamu kirim)
# =========================
AUTO_WEIGHTS = {
    "MobileNetV3": 0.333,
    "EfficientNet": 0.332,
    "DenseNet": 0.335,
}

# =========================
# LOAD MODEL (cache)
# =========================
@st.cache_resource
def load_all_models():
    models = {}
    loaded_count = 0

    for name, filename in MODEL_FILES.items():
        full_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(full_path):
            continue

        try:
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
        except Exception:
            # Recovery kecil
            try:
                models[name] = load_model(full_path, compile=False, custom_objects={"relu6": tf.nn.relu6})
                loaded_count += 1
            except Exception:
                pass

    expected_models = list(MODEL_FILES.keys())
    return models, loaded_count, expected_models


# =========================
# ENSEMBLE INFERENCE
# =========================
def run_research_ensemble(models_dict, weights_dict, image_pil):
    size = (224, 224)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS).convert("RGB")
    img_array_base = np.asarray(image_resized).astype(np.float32)

    model_names = []
    raw_probs_matrix = []
    weighted_sum = np.zeros((N_CLASS,), dtype=np.float32)
    effective_total_weight = 0.0

    for name, model in models_dict.items():
        img_input = img_array_base.copy()
        img_input = PREPROCESS_MAP[name](img_input)
        img_input = np.expand_dims(img_input, axis=0)

        pred_prob = model.predict(img_input, verbose=0)[0]
        w = float(weights_dict.get(name, 0.0))

        model_names.append(name)
        raw_probs_matrix.append(pred_prob)

        weighted_sum += pred_prob * w
        effective_total_weight += w

    if effective_total_weight <= 0:
        effective_total_weight = 1.0

    final_probs = weighted_sum / effective_total_weight
    final_idx = int(np.argmax(final_probs))

    return {
        "final_label": CLASS_NAMES[final_idx],
        "final_conf": float(final_probs[final_idx]),
        "final_probs": final_probs,
        "model_names": model_names,
        "raw_matrix": np.array(raw_probs_matrix),
        "effective_total_weight": effective_total_weight
    }


# =========================
# KNOWLEDGE BASE (RAG) - cache
# =========================
@st.cache_resource
def load_knowledge_base():
    os.makedirs(SOP_FOLDER, exist_ok=True)

    pdf_files = sorted(glob.glob(os.path.join(SOP_FOLDER, "*.pdf")))
    info = {
        "detected": [os.path.basename(p) for p in pdf_files],
        "loaded": [],
        "failed": [],  # list of (filename, reason)
        "pages_loaded": 0,
    }

    if not pdf_files:
        return None, info

    all_documents = []
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Filter halaman kosong (sering terjadi pada PDF scan gambar)
            docs = [d for d in docs if (d.page_content or "").strip()]

            if len(docs) == 0:
                info["failed"].append((fname, "Teks kosong (kemungkinan PDF hasil scan/gambar)"))
                continue

            all_documents.extend(docs)
            info["loaded"].append(fname)
            info["pages_loaded"] += len(docs)

        except Exception as e:
            info["failed"].append((fname, str(e)))

    if not all_documents:
        # PDF ada, tapi semuanya gagal diekstrak teks
        return None, info

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # In-memory Chroma (aman untuk cloud yang kadang read-only)
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        collection_name="kb_jagung"
    )

    return vectorstore, info


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # --- Groq API Key (JANGAN hardcode) ---
    groq_api_key = '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'

    with st.expander("🔐 Groq API Key", expanded=False):
        st.caption("Disarankan pakai **st.secrets** atau ENV, bukan ditulis di source code.")
        if not groq_api_key:
            groq_api_key = 'st.text_input("Masukkan GROQ_API_KEY", type="password")'

    # --- Upload PDF ke knowledge_base ---
    st.markdown("### 📚 Knowledge Base")
    os.makedirs(SOP_FOLDER, exist_ok=True)
    pdf_upload = st.file_uploader("Upload PDF SOP (opsional)", type=["pdf"], accept_multiple_files=True)

    if pdf_upload:
        for f in pdf_upload:
            save_path = os.path.join(SOP_FOLDER, f.name)
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
        st.success("✅ PDF tersimpan. Klik Rebuild KB.")

    if st.button("🔄 Rebuild KB (Clear Cache)"):
        st.cache_resource.clear()
        st.rerun()

    vectorstore_db, kb_info = load_knowledge_base()

    detected_count = len(kb_info["detected"])
    loaded_count_kb = len(kb_info["loaded"])
    failed_count_kb = len(kb_info["failed"])

    st.caption(f"📄 PDF terdeteksi: **{detected_count}**")
    st.caption(f"✅ Berhasil diproses (berteks): **{loaded_count_kb}**")
    st.caption(f"⚠️ Gagal diproses: **{failed_count_kb}**")

    with st.expander("📄 Daftar PDF terdeteksi"):
        if detected_count == 0:
            st.write("Belum ada PDF di folder knowledge_base/.")
        else:
            for f in kb_info["detected"]:
                st.write("• " + f)

    if failed_count_kb > 0:
        with st.expander("⚠️ Detail PDF yang gagal"):
            for fname, reason in kb_info["failed"]:
                st.write(f"❌ {fname} → {reason}")

    st.markdown("---")

    # --- Load models ---
    models_dict, model_ok_count, expected_models = load_all_models()
    missing_models = [m for m in expected_models if m not in models_dict]

    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat!")
        for m in missing_models:
            st.write(f"❌ {m} (cek file: {MODEL_FILES[m]})")
    else:
        st.success(f"✅ Semua {model_ok_count} model berhasil dimuat.")

    st.markdown("---")
    st.markdown("### ⚖️ Bobot Ensemble")

    use_auto = st.checkbox("Gunakan bobot otomatis (dari evaluasi F1)", value=True)

    if use_auto:
        weights_dict = dict(AUTO_WEIGHTS)
        st.info(
            f"Auto → MobileNetV3={weights_dict['MobileNetV3']:.3f}, "
            f"EfficientNet={weights_dict['EfficientNet']:.3f}, "
            f"DenseNet={weights_dict['DenseNet']:.3f}"
        )
    else:
        w_mob = st.number_input("Bobot MobileNetV3", 0.0, 1.0, 0.333, 0.01)
        w_eff = st.number_input("Bobot EfficientNet", 0.0, 1.0, 0.332, 0.01)
        w_dense = st.number_input("Bobot DenseNet", 0.0, 1.0, 0.335, 0.01)
        weights_dict = {"MobileNetV3": w_mob, "EfficientNet": w_eff, "DenseNet": w_dense}

        if st.checkbox("Normalisasi bobot (sum=1)", value=True):
            s = sum(weights_dict.values())
            if s > 0:
                weights_dict = {k: v / s for k, v in weights_dict.items()}


# =========================
# MAIN TABS
# =========================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

# ---------- TAB 1 ----------
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")

    uploaded_file = st.file_uploader("📁 Upload Foto Daun (JPG/PNG)", type=["jpg", "png", "jpeg"])

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

    if image is not None:
        col_img, col_act = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Citra Input", use_container_width=True)

        with col_act:
            st.info("Klik tombol untuk menjalankan inferensi ensemble.")
            if st.button("🔎 Jalankan Inferensi", use_container_width=True):
                if len(models_dict) == 0:
                    st.error("❌ Tidak ada model yang bisa digunakan. Pastikan file `.keras` tersedia.")
                else:
                    with st.spinner("⏳ Sedang menjalankan inferensi..."):
                        result = run_research_ensemble(models_dict, weights_dict, image)
                        st.session_state["diagnosis_result"] = result

    if st.session_state["diagnosis_result"] is not None:
        result = st.session_state["diagnosis_result"]
        st.subheader("✅ Hasil Diagnosis Ensemble")
        st.metric("Prediksi Akhir", result["final_label"], f"{result['final_conf']*100:.2f}%")

        probs_df = pd.DataFrame({
            "Kelas": CLASS_NAMES,
            "Probabilitas": result["final_probs"]
        })

        st.write("**Probabilitas Ensemble (urut sesuai kelas):**")
        st.dataframe(probs_df, use_container_width=True)

        chart = (
            alt.Chart(probs_df)
            .mark_bar()
            .encode(
                x=alt.X("Kelas:N", sort=CLASS_NAMES),
                y=alt.Y("Probabilitas:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Kelas", alt.Tooltip("Probabilitas:Q", format=".4f")]
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        if len(result["model_names"]) > 0:
            st.subheader("🔬 Detail Probabilitas per Model")
            raw_df = pd.DataFrame(result["raw_matrix"], columns=CLASS_NAMES, index=result["model_names"])
            st.dataframe(raw_df, use_container_width=True)
            st.caption(f"Total bobot efektif: **{result['effective_total_weight']:.3f}**")


# ---------- TAB 2 ----------
with tab2:
    st.header("💬 Diskusi Pakar (RAG dari SOP/PDF)")

    detected_count = len(kb_info["detected"])
    loaded_count_kb = len(kb_info["loaded"])

    if detected_count == 0:
        st.warning("Belum ada PDF di folder `knowledge_base/` (atau upload dari sidebar).")
    elif vectorstore_db is None or loaded_count_kb == 0:
        st.error(
            "PDF terdeteksi, tapi **gagal diproses menjadi knowledge base**.\n\n"
            "Penyebab paling sering: PDF hasil scan (tanpa teks), PDF terenkripsi, atau loader gagal baca.\n"
            "Cek sidebar bagian **Detail PDF yang gagal**."
        )
    else:
        st.success("Knowledge base siap. Silakan bertanya.")

    # render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Tanya pakar…")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        if vectorstore_db is None:
            answer = "Knowledge base belum siap. Cek PDF yang gagal diproses di sidebar."
        else:
            retriever = vectorstore_db.as_retriever(search_kwargs={"k": 4})

            # kompatibel untuk versi langchain baru
            try:
                docs = retriever.invoke(user_q)
            except Exception:
                docs = retriever.get_relevant_documents(user_q)

            context = "\n\n".join([d.page_content for d in docs]) if docs else ""

            if not groq_api_key:
                answer = "Groq API Key belum di-set. Isi di sidebar atau set ENV/Secrets."
            else:
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model="llama-3.3-70b-versatile",
                    temperature=0.2
                )

                prompt = f"""
Kamu adalah asisten pakar penyakit daun jagung.
Jawab dengan jelas, ringkas, dan berbasis konteks SOP/PDF berikut.
Jika konteks tidak memuat jawabannya, katakan "tidak ditemukan di SOP".

[KONTEKS]
{context}

[PERTANYAAN]
{user_q}

[JAWABAN]
"""
                try:
                    resp = llm.invoke(prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
                except Exception as e:
                    answer = f"Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
