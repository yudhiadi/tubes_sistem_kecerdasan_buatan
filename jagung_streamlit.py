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
# LANGCHAIN (HANYA GROQ)
# =========================
from langchain_groq import ChatGroq

import traceback

def _is_lfs_pointer(head_bytes: bytes) -> bool:
    try:
        head_text = head_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return "git-lfs.github.com/spec/v1" in head_text

def inspect_model_file(path: str) -> dict:
    info = {
        "path": path,
        "exists": os.path.exists(path),
        "size_mb": None,
        "is_lfs_pointer": False,
        "head_preview": None,
    }
    if not info["exists"]:
        return info

    try:
        size = os.path.getsize(path)
        info["size_mb"] = round(size / (1024 * 1024), 3)
        with open(path, "rb") as f:
            head = f.read(200)
        info["is_lfs_pointer"] = _is_lfs_pointer(head)
        info["head_preview"] = head.decode("utf-8", errors="ignore")[:200]
    except Exception as e:
        info["head_preview"] = f"[inspect error] {e}"
    return info

# Ketergantungan RAG lain dihapus untuk menghindari konflik

# =========================
# KONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Sistem Pakar Jagung (Research Mode)", layout="wide")

if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    # Inisialisasi pesan sambutan
    st.session_state["messages"] = [{"role": "assistant", "content": "Halo! Saya adalah Asisten Pakar Jagung Anda. Anda dapat mengunggah foto daun di tab sebelah, atau bertanya tentang penyakit dan penanganan di sini."}]

# =========================
# PATH AMAN (local + cloud)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    expected_models = list(MODEL_FILES.keys())
    diagnostics = {}

    for name, filename in MODEL_FILES.items():
        full_path = os.path.join(BASE_DIR, filename)

        finfo = inspect_model_file(full_path)
        diagnostics[name] = {
            "file": filename,
            "full_path": full_path,
            "inspect": finfo,
            "load_ok": False,
            "error": None,
            "traceback": None,
        }

        if not finfo["exists"]:
            diagnostics[name]["error"] = "File tidak ditemukan di server."
            continue

        if finfo["is_lfs_pointer"]:
            diagnostics[name]["error"] = "File terdeteksi Git LFS pointer (bukan file model asli)."
            continue

        try:
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
            diagnostics[name]["load_ok"] = True
        except Exception as e1:
            # coba recovery relu6
            try:
                models[name] = load_model(full_path, compile=False, custom_objects={"relu6": tf.nn.relu6})
                loaded_count += 1
                diagnostics[name]["load_ok"] = True
            except Exception as e2:
                diagnostics[name]["error"] = str(e2)
                diagnostics[name]["traceback"] = traceback.format_exc()

    return models, loaded_count, expected_models, diagnostics




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
# KNOWLEDGE BASE (DIPERLUKAN UNTUK KOMPATIBILITAS STRUKTUR SIDEBAR)
# =========================
# Mengganti RAG kompleks dengan dummy function
def load_knowledge_base():
    # Karena RAG dinonaktifkan, kita hanya mengembalikan nilai dummy
    return None, 0, []

# =========================
# PANGGIL MODEL & KB
# =========================
models_dict, model_ok_count, expected_models, model_diag = load_all_models()
vectorstore_db, detected_count, daftar_pdf = load_knowledge_base() 
# ^^^ PENTING: Panggil dummy KB, tangkap 3 nilai agar tidak NameError

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # --- Groq API Key (JANGAN hardcode) ---
    groq_api_key = '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'
    
    # --- Status Model ---
    st.markdown("### 🧠 Status Model")
    st.caption(f"TensorFlow: {tf.__version__}")

    missing_models = [m for m in expected_models if m not in models_dict]

    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat!")
    else:
        st.success(f"✅ Semua {model_ok_count} model berhasil dimuat.")

    for m in expected_models:
        if m in models_dict:
            st.write(f"✅ {m} loaded")
        else:
            st.write(f"❌ {m} gagal")
            with st.expander(f"Detail {m}"):
                st.write(model_diag[m]["full_path"])
                st.write(model_diag[m]["inspect"])
                st.write("Error:", model_diag[m]["error"])
                if model_diag[m]["traceback"]:
                    st.code(model_diag[m]["traceback"])


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

# ---------- TAB 1 (DEEP LEARNING ENSEMBLE) ----------
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")
    st.caption("Unggah foto untuk mendapatkan diagnosis penyakit jagung.")

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


# ---------- TAB 2 (KONSULTASI UMUM VIA GROQ) ----------
with tab2:
    st.header("💬 Diskusi Pakar (Konsultasi AI Umum)")

    # Status berdasarkan diagnosis gambar
    if st.session_state["diagnosis_result"] is not None:
        last_diag = st.session_state["diagnosis_result"]["final_label"]
        st.info(f"💡 **Diagnosis Terakhir:** Sistem mendeteksi **{last_diag}**. Anda dapat menanyakan penanganan penyakit ini.")
        
    else:
        st.info("Silakan unggah foto di tab sebelah atau tanyakan pertanyaan umum tentang budidaya jagung.")

    # render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Tanya pakar…")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Cek apakah ada diagnosis terakhir untuk diberikan sebagai konteks
        if st.session_state["diagnosis_result"] is not None:
            last_diag = st.session_state["diagnosis_result"]["final_label"]
            context_hint = f"Saat menjawab, ingatlah bahwa pengguna baru saja mendiagnosis penyakit: {last_diag}. Berikan saran penanganan yang relevan jika pertanyaannya berkaitan dengan penyakit ini."
        else:
            context_hint = "Tidak ada diagnosis visual yang diberikan. Jawab sebagai pakar pertanian umum."


        if not groq_api_key:
            answer = "Groq API Key belum di-set. Isi di sidebar atau set ENV/Secrets."
        else:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.4 # Naikkan temperature sedikit untuk jawaban yang lebih bervariasi
            )

            prompt = f"""
Anda adalah asisten AI Pakar Penyakit Jagung. Jawablah pertanyaan pengguna dengan profesional, lugas, dan akurat, berdasarkan pengetahuan pertanian umum Anda.

{context_hint}

[PERTANYAAN PENGGUNA]
{user_q}

[JAWABAN AHLI]
"""
            try:
                with st.spinner("AI sedang memikirkan jawaban..."):
                    resp = llm.invoke(prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                answer = f"Error koneksi AI: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)