import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model

# ============================================================
# PREPROCESS INPUT (WAJIB sesuai arsitektur training)
# ============================================================
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep

# ============================================================
# (OPSIONAL) RAG / LangChain
# ============================================================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ============================================================
# 1) KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Sistem Pakar Jagung (Research Mode)",
    layout="wide"
)

# Session state untuk menyimpan hasil inferensi & chat
if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "is_running" not in st.session_state:
    st.session_state["is_running"] = False  # untuk disable tombol saat inferensi jalan

# ============================================================
# 2) LABEL KELAS (URUT SESUAI INDEX OUTPUT MODEL!)
#    (Ini urutan yang kamu minta: 0..3)
# ============================================================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)",   # index 0
    "Karat Daun (Common Rust)",            # index 1
    "Bercak Daun (Gray Leaf Spot)",        # index 2
    "Tanaman Sehat"                        # index 3
]

# ============================================================
# 3) KONFIG MODEL: semua .keras
# ============================================================
MODEL_FILES = {
    "MobileNetV3":  "model_jagung_mobilenetv3_vFinal.keras",
    "EfficientNet": "model_jagung_efficientnet_vFinal.keras",
    "DenseNet":     "model_jagung_densenet_vFinal.keras",
}

PREPROCESS_MAP = {
    "MobileNetV3":  mobilenet_prep,
    "EfficientNet": efficientnet_prep,
    "DenseNet":     densenet_prep,
}

# ============================================================
# 4) BOBOT ENSEMBLE OTOMATIS (dari grafik training kamu)
#    Kenapa pakai inverse val_loss?
#    - val_loss lebih sensitif untuk “confidence” dan generalisasi
#    - makin kecil loss => makin besar bobot
# ============================================================
VAL_LOSS_LAST = {
    "MobileNetV3":  0.163,
    "EfficientNet": 0.148,
    "DenseNet":     0.129,
}

def normalize_weights_from_val_loss(val_loss_dict: dict) -> dict:
    inv = {k: 1.0 / max(v, 1e-8) for k, v in val_loss_dict.items()}  # 1e-8 agar aman dari pembagian nol
    s = sum(inv.values())
    return {k: inv[k] / s for k in inv}

AUTO_WEIGHTS = normalize_weights_from_val_loss(VAL_LOSS_LAST)  # ~ {0.30, 0.33, 0.37}

# ============================================================
# 5) LOAD MODEL (cached supaya tidak load ulang setiap rerun)
# ============================================================
@st.cache_resource
def load_all_models():
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder tempat app.py berada

    loaded_count = 0
    expected = list(MODEL_FILES.keys())

    for name, filename in MODEL_FILES.items():
        full_path = os.path.join(base_dir, filename)

        if not os.path.exists(full_path):
            # Tidak pakai st.error di sini karena fungsi cache (lebih aman print)
            print(f"❌ File tidak ditemukan: {full_path}")
            continue

        try:
            # compile=False:
            # - inferensi saja (lebih cepat)
            # - menghindari mismatch optimizer/loss versi TF
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
        except Exception as e:
            print(f"❌ Gagal load {name}: {e}")

    return models, loaded_count, expected

# ============================================================
# 6) ENSEMBLE INFERENCE (weighted average)
# ============================================================
def run_research_ensemble(models_dict: dict, weights_dict: dict, image_pil: Image.Image):
    # Input size 224x224:
    # - standar ImageNet
    # - konsisten dengan training kamu
    size = (224, 224)

    # ImageOps.fit + LANCZOS:
    # - fit menjaga rasio & crop rapi
    # - LANCZOS kualitas tinggi untuk resize
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS).convert("RGB")
    img_array_base = np.asarray(image_resized).astype(np.float32)

    weighted_sum = np.zeros((len(CLASS_NAMES),), dtype=np.float32)
    effective_total_weight = 0.0

    model_names = []
    raw_probs_matrix = []

    for name, model in models_dict.items():
        # copy agar preprocessing tiap model tidak saling mengganggu
        img_input = img_array_base.copy()

        # preprocess sesuai model
        prep_func = PREPROCESS_MAP.get(name, None)
        if prep_func is not None:
            img_input = prep_func(img_input)

        # bobot model (kalau model tidak ada bobot => 0)
        w = float(weights_dict.get(name, 0.0))

        # shape jadi (1, 224, 224, 3) untuk predict
        img_input = np.expand_dims(img_input, axis=0)

        # output pred_prob = (4,) probabilitas tiap kelas
        pred_prob = model.predict(img_input, verbose=0)[0]

        model_names.append(name)
        raw_probs_matrix.append(pred_prob)

        weighted_sum += pred_prob * w
        effective_total_weight += w

    # safety: kalau semua bobot 0, fallback jadi 1 agar tidak NaN
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
        "effective_total_weight": float(effective_total_weight),
    }

# ============================================================
# 7) KNOWLEDGE BASE (RAG) - cached
# ============================================================
@st.cache_resource
def load_knowledge_base():
    SOP_FOLDER = "knowledge_base"
    os.makedirs(SOP_FOLDER, exist_ok=True)  # otomatis buat folder jika belum ada

    pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    if not pdf_files:
        return None, 0

    all_docs = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] gagal load PDF {pdf_path}: {e}")

    if not all_docs:
        return None, 0

    # chunk_size=1000 & overlap=200:
    # - 1000 karakter cukup buat konteks teknis SOP
    # - overlap 200 menjaga konteks nyambung antar chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)

    return vectorstore, len(pdf_files)

vectorstore_db, jumlah_pdf = load_knowledge_base()

# ============================================================
# 8) SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # Jangan hardcode API key di kode publik.
    # Simpan di .streamlit/secrets.toml atau ENV.
    # secrets.toml:
    # GROQ_API_KEY="xxxxx"
    groq_api_key = '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'

    models_dict, count, expected_models = load_all_models()
    missing_models = [m for m in expected_models if m not in models_dict]

    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat:")
        for m in missing_models:
            st.write(f"❌ **{m}** (cek file `{MODEL_FILES[m]}`)")
        st.caption("Pastikan semua file `.keras` ada satu folder dengan `app.py`.")
    else:
        st.success(f"✅ Semua {count} model berhasil dimuat.")

    st.markdown("### ⚖️ Bobot Ensemble (Otomatis dari training)")
    st.write("Menggunakan inverse **validation loss** (lebih kecil loss → bobot lebih besar).")

    # Default bobot otomatis
    default_eff = float(AUTO_WEIGHTS["EfficientNet"])
    default_den = float(AUTO_WEIGHTS["DenseNet"])
    default_mob = float(AUTO_WEIGHTS["MobileNetV3"])

    # Jika mau dikunci otomatis, kamu bisa hapus checkbox + slider override di bawah.
    allow_override = st.checkbox("Override manual bobot", value=False)

    if allow_override:
        w_eff = st.slider("Bobot EfficientNet", 0.0, 1.0, default_eff, 0.01)
        w_den = st.slider("Bobot DenseNet", 0.0, 1.0, default_den, 0.01)
        w_mob = st.slider("Bobot MobileNetV3", 0.0, 1.0, default_mob, 0.01)
        # Normalisasi supaya total bobot = 1 (lebih stabil untuk interpretasi)
        s = (w_eff + w_den + w_mob) if (w_eff + w_den + w_mob) > 0 else 1.0
        weights_dict = {
            "EfficientNet": w_eff / s,
            "DenseNet": w_den / s,
            "MobileNetV3": w_mob / s,
        }
    else:
        weights_dict = {
            "EfficientNet": default_eff,
            "DenseNet": default_den,
            "MobileNetV3": default_mob,
        }

    st.caption(
        f"Bobot aktif: "
        f"Eff={weights_dict['EfficientNet']:.2f}, "
        f"Den={weights_dict['DenseNet']:.2f}, "
        f"Mob={weights_dict['MobileNetV3']:.2f}"
    )

    st.markdown("---")
    st.caption(f"📚 Knowledge base PDF terdeteksi: {jumlah_pdf}")

# ============================================================
# 9) MAIN TABS
# ============================================================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

# ============================================================
# TAB 1: INFERENSI
# ============================================================
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")

    # Pilih sumber gambar: upload atau kamera
    source = st.radio("Sumber gambar", ["Upload File", "Kamera"], horizontal=True)

    image = None

    if source == "Upload File":
        uploaded = st.file_uploader("Upload Sampel Daun", type=["jpg", "png", "jpeg"])
        if uploaded:
            image = Image.open(uploaded)

    else:
        # st.camera_input memungkinkan ambil foto dari kamera (browser/HP)
        cam = st.camera_input("Ambil foto daun")
        if cam:
            image = Image.open(cam)

    if image is not None:
        col_img, col_act = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Citra Input", use_container_width=True)

        with col_act:
            st.info("Klik tombol untuk menjalankan inferensi ensemble.")
            clicked = st.button(
                "🔎 Jalankan Inferensi",
                use_container_width=True,
                disabled=st.session_state["is_running"]
            )

            if clicked:
                if len(models_dict) == 0:
                    st.error("❌ Tidak ada model yang bisa digunakan. Pastikan file `.keras` tersedia.")
                else:
                    st.session_state["is_running"] = True

                    # Spinner = penanda loading yang kamu minta
                    with st.spinner("Sedang menjalankan inferensi..."):
                        result = run_research_ensemble(models_dict, weights_dict, image)

                    st.session_state["diagnosis_result"] = result
                    st.session_state["is_running"] = False

    # ===========================
    # TAMPILKAN HASIL
    # ===========================
    if st.session_state["diagnosis_result"] is not None:
        result = st.session_state["diagnosis_result"]

        st.subheader("✅ Hasil Diagnosis Ensemble")
        st.metric("Prediksi Akhir", result["final_label"], f"{result['final_conf']*100:.2f}%")

        # DataFrame probabilitas: JANGAN di-sort agar urut sesuai index 0..3
        probs_df = pd.DataFrame({
            "Kelas": CLASS_NAMES,                 # urut tetap
            "Probabilitas": result["final_probs"] # urut tetap
        })

        st.write("**Probabilitas Ensemble (Weighted Average) — urut sesuai index kelas:**")
        st.dataframe(probs_df, use_container_width=True)

        # Chart juga urut sesuai CLASS_NAMES (tanpa sort)
        chart = alt.Chart(probs_df).mark_bar().encode(
            x=alt.X("Kelas:N", sort=CLASS_NAMES),  # paksa urutan kategori sesuai list
            y=alt.Y("Probabilitas:Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Kelas", alt.Tooltip("Probabilitas:Q", format=".4f")]
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)

        # Detail per model
        if len(result["model_names"]) > 0:
            st.subheader("🔬 Detail Probabilitas per Model")
            raw_df = pd.DataFrame(
                result["raw_matrix"],
                columns=CLASS_NAMES,
                index=result["model_names"]
            )
            st.dataframe(raw_df, use_container_width=True)
            st.caption(f"Total bobot efektif: **{result['effective_total_weight']:.2f}**")

# ============================================================
# TAB 2: CHATBOT (RAG)
# ============================================================
with tab2:
    st.header("💬 Diskusi Pakar (RAG dari SOP/PDF)")

    if jumlah_pdf == 0 or vectorstore_db is None:
        st.warning("Belum ada PDF di folder `knowledge_base/`.")
    else:
        st.success("Knowledge base siap.")

    # tampilkan history chat
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
            docs = retriever.get_relevant_documents(user_q)
            context = "\n\n".join([d.page_content for d in docs])

            if not groq_api_key:
                answer = "Groq API Key belum di-set (pakai secrets.toml atau ENV)."
            else:
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
