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
# PREPROCESS INPUT (WAJIB sesuai backbone)
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
# 1) KONFIGURASI STREAMLIT
# =========================
st.set_page_config(
    page_title="Sistem Pakar Jagung (Research Mode)",
    layout="wide"
)

# Simpan state agar hasil inferensi & chat tidak hilang saat rerun
if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# =========================
# 2) LABEL KELAS (URUTAN WAJIB)
# =========================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)",  # index 0
    "Karat Daun (Common Rust)",           # index 1
    "Bercak Daun (Gray Leaf Spot)",       # index 2
    "Tanaman Sehat"                       # index 3
]

N_CLASS = len(CLASS_NAMES)  # 4 kelas

# =========================
# 3) MODEL FILES (.keras semua)
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
# 4) BOBOT AUTO (dari grafik F1 per kelas yang kamu kirim)
#    - Hitung dari rata-rata F1 (macro over classes)
# =========================
AUTO_WEIGHTS = {
    "MobileNetV3": 0.333,
    "EfficientNet": 0.332,
    "DenseNet": 0.335,
}
# Catatan: sudah dinormalisasi (jumlahnya ~ 1.0)

# =========================
# 5) LOAD MODEL (cache)
# =========================
@st.cache_resource
def load_all_models():
    """
    Load semua model dari folder yang sama dengan file Streamlit ini.
    cache_resource => model tidak reload setiap rerun UI.
    """
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))

    loaded_count = 0
    for name, filename in MODEL_FILES.items():
        full_path = os.path.join(base_dir, filename)

        if not os.path.exists(full_path):
            # jangan crash: cukup catat missing
            print(f"❌ File tidak ditemukan: {full_path}")
            continue

        try:
            # Normal case: .keras harusnya aman compile=False
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
        except Exception as e:
            # Recovery kecil untuk beberapa kasus custom op
            print(f"⚠️ Gagal load {name} normal: {e}")
            try:
                models[name] = load_model(
                    full_path,
                    compile=False,
                    custom_objects={"relu6": tf.nn.relu6}
                )
                loaded_count += 1
                print(f"✅ Recovery berhasil untuk {name}")
            except Exception as e2:
                print(f"❌ Recovery gagal {name}: {e2}")

    expected_models = list(MODEL_FILES.keys())
    return models, loaded_count, expected_models

# =========================
# 6) ENSEMBLE INFERENCE (weighted average)
# =========================
def run_research_ensemble(models_dict, weights_dict, image_pil):
    """
    Weighted average:
      final_probs = sum_i (w_i * p_i) / sum_i (w_i)
    - p_i: probabilitas softmax model i (shape [4])
    - w_i: bobot model i
    """

    # Samakan size input: 224x224 (umum untuk ImageNet backbones)
    size = (224, 224)

    # Fit+crop biar rasio tetap bagus (daripada stretch)
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS).convert("RGB")

    # Ubah ke array float32 (TF friendly)
    img_array_base = np.asarray(image_resized).astype(np.float32)

    model_names = []
    raw_probs_matrix = []
    weighted_sum = np.zeros((N_CLASS,), dtype=np.float32)
    effective_total_weight = 0.0

    for name, model in models_dict.items():
        # Copy array mentah, lalu preprocess sesuai backbone model tsb
        img_input = img_array_base.copy()
        img_input = PREPROCESS_MAP[name](img_input)

        # Tambah dimensi batch: (1, 224, 224, 3)
        img_input = np.expand_dims(img_input, axis=0)

        # Prediksi => (1,4) lalu ambil [0]
        pred_prob = model.predict(img_input, verbose=0)[0]

        # Bobot model (kalau 0, kontribusinya 0)
        w = float(weights_dict.get(name, 0.0))

        model_names.append(name)
        raw_probs_matrix.append(pred_prob)

        weighted_sum += pred_prob * w
        effective_total_weight += w

    # Kalau user set semua bobot 0, jangan bagi 0
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
# 7) KNOWLEDGE BASE (RAG) - cache
# =========================
@st.cache_resource
def load_knowledge_base():
    """
    Load semua PDF di folder knowledge_base/ lalu buat vectorstore Chroma.
    cache_resource => tidak rebuild embedding setiap rerun.
    """
    SOP_FOLDER = "knowledge_base"
    os.makedirs(SOP_FOLDER, exist_ok=True)

    pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    if not pdf_files:
        return None, 0

    all_documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"[WARN] gagal load PDF {pdf_path}: {e}")

    if not all_documents:
        return None, len(pdf_files)

    # chunk_size=1000 cukup “padat”, chunk_overlap=200 biar konteks nyambung
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # persist_directory biar lebih stabil (opsional, tapi enak untuk app)
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="kb_jagung"
    )

    return vectorstore, len(pdf_files)

vectorstore_db, jumlah_pdf = load_knowledge_base()

# =========================
# 8) SIDEBAR UI
# =========================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # Jangan hardcode API key di source code (lebih aman pakai st.secrets / env)
    groq_api_key = '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'


    models_dict, count, expected_models = load_all_models()

    missing_models = [m for m in expected_models if m not in models_dict]
    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat!")
        for m in missing_models:
            st.write(f"❌ **{m}** (cek file: {MODEL_FILES[m]})")
        st.caption("Pastikan semua file `.keras` berada 1 folder dengan script Streamlit.")
    else:
        st.success(f"✅ Semua {count} model berhasil dimuat.")

    st.markdown("---")
    st.markdown("### ⚖️ Bobot Ensemble")

    use_auto = st.checkbox("Gunakan bobot otomatis (dari evaluasi F1)", value=True)

    if use_auto:
        weights_dict = dict(AUTO_WEIGHTS)
        st.info(
            f"Auto Weights → MobileNetV3={weights_dict['MobileNetV3']:.3f}, "
            f"EfficientNet={weights_dict['EfficientNet']:.3f}, "
            f"DenseNet={weights_dict['DenseNet']:.3f}"
        )
    else:
        # Manual mode
        w_mob = st.number_input("Bobot MobileNetV3", 0.0, 1.0, 0.333, 0.01)
        w_eff = st.number_input("Bobot EfficientNet", 0.0, 1.0, 0.332, 0.01)
        w_dense = st.number_input("Bobot DenseNet", 0.0, 1.0, 0.335, 0.01)

        weights_dict = {"MobileNetV3": w_mob, "EfficientNet": w_eff, "DenseNet": w_dense}

        # Optional: normalisasi supaya sum=1 (lebih “rapi”)
        if st.checkbox("Normalisasi bobot (sum=1)", value=True):
            s = sum(weights_dict.values())
            if s > 0:
                weights_dict = {k: v / s for k, v in weights_dict.items()}

    st.markdown("---")
    st.caption(f"📚 Knowledge base PDF terdeteksi: {jumlah_pdf}")

# =========================
# 9) MAIN TABS
# =========================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

# ---------- TAB 1 ----------
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")

    # Upload & Kamera
    colA, colB = st.columns(2)

    with colA:
        uploaded_file = st.file_uploader("📁 Upload Sampel Daun", type=["jpg", "png", "jpeg"])

    with colB:
        cam_file = st.camera_input("📷 Ambil Foto dari Kamera")

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif cam_file is not None:
        image = Image.open(io.BytesIO(cam_file.getvalue()))

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
                    # LOADING INDICATOR
                    with st.spinner("⏳ Sedang menjalankan inferensi..."):
                        result = run_research_ensemble(models_dict, weights_dict, image)
                        st.session_state["diagnosis_result"] = result

    # Tampilkan hasil
    if st.session_state["diagnosis_result"] is not None:
        result = st.session_state["diagnosis_result"]

        st.subheader("✅ Hasil Diagnosis Ensemble")
        st.metric("Prediksi Akhir", result["final_label"], f"{result['final_conf']*100:.2f}%")

        # Probabilitas harus URUT sesuai CLASS_NAMES (jangan di-sort)
        probs_df = pd.DataFrame({
            "Kelas": CLASS_NAMES,
            "Probabilitas": result["final_probs"]
        })

        st.write("**Probabilitas Ensemble (Weighted Average) — urut sesuai kelas:**")
        st.dataframe(probs_df, use_container_width=True)

        # Grafik Altair: kunci sort sesuai CLASS_NAMES
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

        # Detail probabilitas per model
        if len(result["model_names"]) > 0:
            st.subheader("🔬 Detail Probabilitas per Model")
            raw_df = pd.DataFrame(
                result["raw_matrix"],
                columns=CLASS_NAMES,
                index=result["model_names"]
            )
            st.dataframe(raw_df, use_container_width=True)
            st.caption(f"Total bobot efektif: **{result['effective_total_weight']:.3f}**")

# ---------- TAB 2 ----------
with tab2:
    st.header("💬 Diskusi Pakar (RAG dari SOP/PDF)")

    if jumlah_pdf == 0 or vectorstore_db is None:
        st.warning("Belum ada PDF di folder `knowledge_base/`.")
    else:
        st.success("Knowledge base siap.")

    # Render history chat
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
        else:
            retriever = vectorstore_db.as_retriever(search_kwargs={"k": 4})

            # FIX: versi LangChain baru umumnya pakai invoke()
            try:
                docs = retriever.invoke(user_q)
            except Exception:
                # fallback (kalau ada versi lama)
                docs = retriever.get_relevant_documents(user_q)

            context = "\n\n".join([d.page_content for d in docs]) if docs else ""

            if not groq_api_key:
                answer = "Groq API Key belum di-set."
            else:
                # Ganti model Groq yang masih aktif
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model="llama-3.3-70b-versatile",  # alternatif ringan: "llama-3.1-8b-instant"
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
