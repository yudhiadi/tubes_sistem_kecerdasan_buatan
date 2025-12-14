import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# ============================================================
# PREPROCESS INPUT (WAJIB sesuai arsitektur)
# ============================================================
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep

# ============================================================
# LANGCHAIN (RAG)
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ============================================================
# 1) KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Sistem Pakar Jagung (Research Mode)",
    layout="wide"
)

# Simpan hasil inferensi & chat history di session_state agar tidak hilang saat rerun
if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ============================================================
# 2) LABEL KELAS (sesuaikan dengan training)
# ============================================================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)",  # index 0
    "Karat Daun (Common Rust)",           # index 1
    "Bercak Daun (Gray Leaf Spot)",       # index 2
    "Tanaman Sehat"                       # index 3
]

N_CLASSES = len(CLASS_NAMES)  # agar tidak hardcode "4" di banyak tempat

# ============================================================
# 3) LOAD MODEL (SEMUA .keras)
# ============================================================
@st.cache_resource
def load_all_models():
    """
    Load semua model dari folder yang sama dengan app.py.
    Kenapa @st.cache_resource?
    - Model berat, kalau di-load setiap rerun akan lambat.
    - Cache membuat model hanya di-load sekali selama session.
    """
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder app.py

    # Semua file model pakai .keras (lebih robust dari .h5)
    model_files = {
        "EfficientNet": "model_jagung_efficientnet_vFinal.keras",
        "DenseNet":     "model_jagung_densenet_vFinal.keras",
        "MobileNetV3":  "model_jagung_mobilenetv3_vFinal.keras",
    }

    loaded_count = 0
    for name, filename in model_files.items():
        full_path = os.path.join(base_dir, filename)

        if not os.path.exists(full_path):
            # File tidak ada -> nanti muncul di daftar missing_models
            print(f"❌ File tidak ditemukan: {full_path}")
            continue

        try:
            # compile=False: inference only (lebih cepat dan minim masalah optimizer/loss)
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
        except Exception as e:
            # Jika gagal load (versi TF beda / file corrupt / custom layer)
            print(f"❌ Gagal load {name}: {e}")

    return models, loaded_count, list(model_files.keys())

# ============================================================
# 4) NORMALISASI BOBOT ENSEMBLE (PENTING)
# ============================================================
def normalize_weights(raw_weights: dict, available_models: list):
    """
    Menormalkan bobot agar total = 1, dan hanya untuk model yang benar-benar tersedia (loaded).
    Kenapa perlu?
    - Weighted average harus pakai bobot yang konsisten (sum=1) agar probabilitas final valid.
    - Kalau ada model gagal load, bobotnya harus dibuang lalu dinormalisasi ulang.
    """
    # Ambil bobot hanya untuk model yang ada
    w = {m: float(raw_weights.get(m, 0.0)) for m in available_models}

    # Buang nilai negatif (defensive)
    for k in w:
        if w[k] < 0:
            w[k] = 0.0

    total = sum(w.values())

    # Jika user set semua 0, fallback: rata untuk model yang tersedia
    if total <= 0 and len(available_models) > 0:
        equal = 1.0 / len(available_models)
        return {m: equal for m in available_models}, total

    # Normalisasi agar sum=1
    return {m: (w[m] / total) for m in available_models}, total

# ============================================================
# 5) ENSEMBLE INFERENCE (WEIGHTED AVERAGE)
# ============================================================
def run_research_ensemble(models_dict, weights_norm, image_pil):
    """
    Weighted average ensemble:
    final_probs = Σ (prob_model_i * weight_i)
    """
    size = (224, 224)  # 224x224 umum untuk ImageNet dan konsisten dengan training kamu

    # exif_transpose: fix foto dari HP yang sering “kebalik” orientasi
    image_pil = ImageOps.exif_transpose(image_pil)

    # Fit + resize tanpa distorsi berlebihan (crop/fit), lebih stabil untuk input model
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS).convert("RGB")

    img_array_base = np.asarray(image_resized).astype(np.float32)

    model_names = []
    raw_probs_matrix = []
    weighted_sum = np.zeros((N_CLASSES,), dtype=np.float32)

    # Loop hanya model yang berhasil di-load
    for name, model in models_dict.items():
        img_input = img_array_base.copy()

        # Preprocess sesuai arsitektur (kalau salah -> skor drop drastis)
        if name == "MobileNetV3":
            img_input = mobilenet_prep(img_input)
        elif name == "DenseNet":
            img_input = densenet_prep(img_input)
        elif name == "EfficientNet":
            img_input = efficientnet_prep(img_input)

        # Tambah dimensi batch: (H,W,C) -> (1,H,W,C)
        img_input = np.expand_dims(img_input, axis=0)

        # Probabilitas softmax: shape (N_CLASSES,)
        pred_prob = model.predict(img_input, verbose=0)[0]

        # Bobot model ini (sudah ternormalisasi)
        w = float(weights_norm.get(name, 0.0))

        model_names.append(name)
        raw_probs_matrix.append(pred_prob)

        # Weighted sum
        weighted_sum += pred_prob * w

    final_probs = weighted_sum
    final_idx = int(np.argmax(final_probs))

    return {
        "final_label": CLASS_NAMES[final_idx],
        "final_conf": float(final_probs[final_idx]),
        "final_probs": final_probs,
        "model_names": model_names,
        "raw_matrix": np.array(raw_probs_matrix),
        "weights_norm": weights_norm
    }

# ============================================================
# 6) KNOWLEDGE BASE (RAG) dari PDF SOP
# ============================================================
@st.cache_resource
def load_knowledge_base():
    SOP_FOLDER = "knowledge_base"  # folder relatif (sejajar app.py)
    os.makedirs(SOP_FOLDER, exist_ok=True)

    pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    if not pdf_files:
        return None, 0

    all_documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"[WARN] gagal load PDF {pdf_path}: {e}")

    if not all_documents:
        return None, 0

    # chunk_size=1000 & overlap=200: umum untuk RAG agar konteks cukup besar tapi tidak kepanjangan
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # In-memory Chroma (cukup untuk riset). Jika mau persist, bisa tambah persist_directory.
    vectorstore = Chroma.from_documents(splits, embeddings)

    return vectorstore, len(pdf_files)

vectorstore_db, jumlah_pdf = load_knowledge_base()

# ============================================================
# 7) SIDEBAR UI
# ============================================================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # ⚠️ Lebih aman: simpan API key di st.secrets, tapi ini tetap jalan sesuai kode kamu
    groq_api_key = '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'

    # Load model
    models_dict, count, expected_models = load_all_models()

    # Alert model hilang / gagal load
    missing_models = [m for m in expected_models if m not in models_dict]
    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat!")
        for m in missing_models:
            st.write(f"❌ **{m}** (file .keras tidak ditemukan / gagal load)")
        st.caption("Pastikan file .keras ada di folder yang sama dengan app.py.")
    else:
        st.success(f"✅ Semua {count} model berhasil dimuat.")

    st.markdown("---")
    st.markdown("### ⚖️ Konfigurasi Bobot Ensemble")

    # number_input: step 0.05/0.1 mudah diatur saat demo
    w_eff  = st.number_input("Bobot EfficientNet", 0.0, 1.0, 0.4, 0.1)
    w_dense= st.number_input("Bobot DenseNet",     0.0, 1.0, 0.4, 0.1)
    w_mob  = st.number_input("Bobot MobileNetV3",  0.0, 1.0, 0.2, 0.1)

    raw_weights = {"EfficientNet": w_eff, "DenseNet": w_dense, "MobileNetV3": w_mob}

    # Normalisasi bobot berdasarkan model yang benar-benar tersedia
    available_models = list(models_dict.keys())
    weights_norm, sum_raw = normalize_weights(raw_weights, available_models)

    st.caption(f"Σ bobot input (raw): **{sum_raw:.2f}**")
    st.write("**Bobot ternormalisasi (dipakai untuk ensemble):**")
    st.json(weights_norm)

    st.markdown("---")
    st.caption(f"📚 Knowledge base PDF terdeteksi: {jumlah_pdf}")

# ============================================================
# 8) MAIN CONTENT
# ============================================================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

# ----------------------------
# TAB 1: INFERENSI (UPLOAD / KAMERA)
# ----------------------------
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")

    # Pilih sumber input gambar
    source = st.radio("Sumber gambar", ["Upload File", "Kamera"], horizontal=True)

    image = None

    if source == "Upload File":
        uploaded_file = st.file_uploader("Upload sampel daun", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        cam_file = st.camera_input("Ambil foto dari kamera")
        if cam_file is not None:
            image = Image.open(cam_file)

    if image is not None:
        col_img, col_act = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Citra Input", use_container_width=True)

        with col_act:
            st.info("Klik tombol untuk menjalankan inferensi ensemble.")
            if st.button("🔎 Jalankan Inferensi", use_container_width=True):
                if len(models_dict) == 0:
                    st.error("❌ Tidak ada model yang bisa digunakan. Pastikan file .keras tersedia.")
                else:
                    result = run_research_ensemble(models_dict, weights_norm, image)
                    st.session_state["diagnosis_result"] = result

    if st.session_state["diagnosis_result"] is not None:
        result = st.session_state["diagnosis_result"]

        st.subheader("✅ Hasil Diagnosis Ensemble")
        st.metric("Prediksi Akhir", result["final_label"], f"{result['final_conf']*100:.2f}%")

        probs_df = pd.DataFrame({
            "Kelas": CLASS_NAMES,
            "Probabilitas": result["final_probs"]
        }).sort_values("Probabilitas", ascending=False)

        st.write("**Probabilitas Ensemble (Weighted Average):**")
        st.dataframe(probs_df, use_container_width=True)

        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X("Kelas:N", sort="-y"),
            y=alt.Y("Probabilitas:Q"),
            tooltip=["Kelas", alt.Tooltip("Probabilitas:Q", format=".3f")]
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)

        # Detail probabilitas per model
        if len(result["model_names"]) > 0:
            st.subheader("🔬 Detail Probabilitas per Model")
            raw_df = pd.DataFrame(
                result["raw_matrix"],
                columns=CLASS_NAMES,
                index=result["model_names"]
            )
            st.dataframe(raw_df, use_container_width=True)
            st.caption(f"Bobot ternormalisasi: **{result['weights_norm']}**")

# ----------------------------
# TAB 2: CHATBOT (RAG)
# ----------------------------
with tab2:
    st.header("💬 Diskusi Pakar (RAG dari SOP/PDF)")

    if jumlah_pdf == 0 or vectorstore_db is None:
        st.warning("Belum ada PDF di folder `knowledge_base/`.")
    else:
        st.success("Knowledge base siap.")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Tanya pakar…")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        if vectorstore_db is None:
            answer = "Knowledge base belum tersedia."
        elif not groq_api_key:
            answer = "Groq API Key belum di-set."
        else:
            retriever = vectorstore_db.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(user_q)
            context = "\n\n".join([d.page_content for d in docs])

            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.1-70b-versatile",
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
