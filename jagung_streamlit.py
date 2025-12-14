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
# PREPROCESS INPUT (WAJIB SESUAI ARSITEKTUR)
# ============================================================
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep

# ============================================================
# LANGCHAIN (RAG dari PDF SOP)
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

# Simpan hasil prediksi agar tidak hilang saat UI rerun
if "diagnosis_result" not in st.session_state:
    st.session_state["diagnosis_result"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ============================================================
# 2) LABEL KELAS (URUTAN WAJIB SESUAI TRAINING)
# ============================================================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)",  # index 0
    "Karat Daun (Common Rust)",           # index 1
    "Bercak Daun (Gray Leaf Spot)",       # index 2
    "Tanaman Sehat"                       # index 3
]
N_CLASSES = len(CLASS_NAMES)  # 4 kelas

# ============================================================
# 3) LOAD MODEL (.KERAS SEMUA) + PREPROCESS MAP
# ============================================================
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

# ============================================================
# 3A) OPTIONAL: AUTO WEIGHT DARI FILE REPORT (macro avg f1)
#    - Kalau file tidak ada, fallback ke bobot manual slider
# ============================================================
def _parse_macro_f1_from_report(txt_path: str) -> float | None:
    """
    Ambil macro avg f1-score dari file classification_report.txt.
    Kenapa macro avg?
    - Lebih adil untuk dataset multi-kelas (tiap kelas bobot sama).
    - Cocok jadi dasar pembobotan ensemble sederhana.
    """
    if not os.path.exists(txt_path):
        return None
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        # cari baris yang mengandung "macro avg"
        for line in lines:
            if "macro avg" in line:
                parts = line.split()
                # format sklearn umumnya: macro avg  precision  recall  f1-score  support
                # index terakhir ke-2 biasanya f1-score
                # contoh: ["macro","avg","0.93","0.92","0.92","1234"]
                # f1-score ada di parts[4]
                if len(parts) >= 5:
                    return float(parts[4])
    except Exception:
        return None
    return None

def compute_auto_weights() -> dict:
    """
    Hitung bobot otomatis dari macro F1 tiap model.
    - Normalisasi: w_i = f1_i / sum(f1)
    - Jika semua gagal terbaca -> fallback bobot rata (1/3).
    """
    scores = {}
    for name in MODEL_FILES.keys():
        # nama file report boleh kamu sesuaikan dengan naming-mu
        # contoh kamu sebelumnya: MobileNetV3_classification_report.txt
        report_guess = f"{name}_classification_report.txt"
        f1 = _parse_macro_f1_from_report(report_guess)
        if f1 is not None and f1 > 0:
            scores[name] = f1

    if not scores:
        # fallback rata jika tidak ada report
        return {k: 1.0 / len(MODEL_FILES) for k in MODEL_FILES.keys()}

    s = sum(scores.values())
    # untuk model yang tidak punya skor, kita kasih 0 supaya tidak ikut “menarik” prediksi
    weights = {k: (scores.get(k, 0.0) / s) for k in MODEL_FILES.keys()}
    return weights

@st.cache_resource
def load_all_models():
    """
    Load semua model .keras dari folder yang sama dengan file app.py.
    Kenapa pakai cache_resource?
    - Load model itu berat, jangan diulang tiap rerun Streamlit.
    """
    models = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    expected = list(MODEL_FILES.keys())

    loaded_count = 0
    for name in expected:
        filename = MODEL_FILES[name]
        full_path = os.path.join(base_dir, filename)

        if not os.path.exists(full_path):
            continue

        try:
            # compile=False: inferensi saja -> lebih cepat & minim mismatch optimizer/loss
            models[name] = load_model(full_path, compile=False)
            loaded_count += 1
        except TypeError:
            # fallback jika ada custom object (jarang untuk .keras, tapi aman)
            models[name] = load_model(
                full_path,
                compile=False,
                custom_objects={"relu6": tf.nn.relu6}
            )
            loaded_count += 1
        except Exception:
            # biarkan model ini gagal, UI akan memberi alert
            pass

    return models, loaded_count, expected

# ============================================================
# 4) ENSEMBLE INFERENCE (WEIGHTED AVERAGE)
# ============================================================
def run_research_ensemble(models_dict, weights_dict, image_pil):
    """
    - Resize ke 224x224 (standar ImageNet; konsisten dengan training kamu)
    - Preprocess sesuai arsitektur
    - Probabilitas final = sum(w_i * p_i) / sum(w_i)
    """
    size = (224, 224)  # 224: input standar MobileNet/EfficientNet/DenseNet dari ImageNet
    image_resized = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS)  # LANCZOS: resize kualitas bagus
    image_resized = image_resized.convert("RGB")  # pastikan 3 channel
    img_array_base = np.asarray(image_resized).astype(np.float32)  # float32: standar input TF

    model_names = []
    raw_probs_matrix = []
    weighted_sum = np.zeros((N_CLASSES,), dtype=np.float32)
    effective_total_weight = 0.0

    for name, model in models_dict.items():
        # kalau output kelas tidak cocok, skip (defensive)
        try:
            out_dim = int(model.output_shape[-1])
            if out_dim != N_CLASSES:
                continue
        except Exception:
            pass

        img_input = img_array_base.copy()

        # preprocess sesuai arsitektur -> mencegah skor drop karena skala input salah
        prep_func = PREPROCESS_MAP.get(name, None)
        if prep_func is not None:
            img_input = prep_func(img_input)

        # ambil bobot model (pastikan key sama: MobileNetV3/EfficientNet/DenseNet)
        w = float(weights_dict.get(name, 0.0))

        img_input = np.expand_dims(img_input, axis=0)  # shape jadi (1,224,224,3)

        # pred_prob shape: (4,)
        pred_prob = model.predict(img_input, verbose=0)[0]

        model_names.append(name)
        raw_probs_matrix.append(pred_prob)

        weighted_sum += pred_prob * w
        effective_total_weight += w

    # kalau semua bobot 0, jangan bagi 0 -> paksa jadi 1
    if effective_total_weight == 0:
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
        "weights_used": {k: float(weights_dict.get(k, 0.0)) for k in MODEL_FILES.keys()}
    }

# ============================================================
# 5) KNOWLEDGE BASE (RAG) - PDF SOP
# ============================================================
@st.cache_resource
def load_knowledge_base():
    """
    Load PDF dari folder knowledge_base/ lalu buat vectorstore.
    - chunk_size=1000 & overlap=200: umum untuk balancing konteks vs recall.
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
        except Exception:
            pass

    if not all_documents:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore, len(pdf_files)

vectorstore_db, jumlah_pdf = load_knowledge_base()

# ============================================================
# 6) SIDEBAR UI (MODEL STATUS + WEIGHTS)
# ============================================================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Models: MobileNetV3, EfficientNet, DenseNet")
    st.markdown("---")

    # ⚠️ Jangan hardcode API key di kode publik.
    # Taruh di .streamlit/secrets.toml lalu ambil: st.secrets["GROQ_API_KEY"]
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    models_dict, count, expected_models = load_all_models()

    missing_models = [m for m in expected_models if m not in models_dict]
    if missing_models:
        st.error("⚠️ Beberapa model gagal dimuat!")
        for m in missing_models:
            st.write(f"❌ **{m}** (cek file: `{MODEL_FILES[m]}`)")
        st.caption("Pastikan semua file `.keras` ada di folder yang sama dengan `app.py`.")
    else:
        st.success(f"✅ Semua {count} model berhasil dimuat.")

    st.markdown("---")

    st.markdown("### ⚖️ Bobot Ensemble")
    use_auto = st.toggle("Gunakan bobot otomatis (dari macro F1 report jika ada)", value=True)

    if use_auto:
        auto_w = compute_auto_weights()
        st.info("Bobot otomatis dinormalisasi (jumlah = 1).")
        st.write(auto_w)

        weights_dict = auto_w
    else:
        # Bobot manual (range 0..1), step 0.05 biar gampang adjust
        w_mob = st.slider("Bobot MobileNetV3", 0.0, 1.0, 0.34, 0.05)
        w_eff = st.slider("Bobot EfficientNet", 0.0, 1.0, 0.33, 0.05)
        w_den = st.slider("Bobot DenseNet", 0.0, 1.0, 0.33, 0.05)

        weights_dict = {"MobileNetV3": w_mob, "EfficientNet": w_eff, "DenseNet": w_den}

        # Normalisasi supaya total bobot efektif = 1 (lebih stabil & mudah dipahami)
        s = sum(weights_dict.values())
        if s == 0:
            st.warning("Semua bobot = 0. Sistem akan fallback pembagi = 1 saat inferensi.")
        else:
            weights_dict = {k: v / s for k, v in weights_dict.items()}
            st.caption("Bobot dinormalisasi otomatis (jumlah = 1).")
            st.write(weights_dict)

    st.markdown("---")
    st.caption(f"📚 Knowledge base PDF terdeteksi: {jumlah_pdf}")

# ============================================================
# 7) MAIN CONTENT
# ============================================================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

# ============================================================
# TAB 1: INFERENSI (UPLOAD + CAMERA INPUT)
# ============================================================
with tab1:
    st.header("Analisis Citra (Deep Learning Ensemble)")

    # Pilih sumber gambar: upload atau kamera
    src = st.radio("Sumber gambar", ["Upload File", "Kamera"], horizontal=True)

    image = None

    if src == "Upload File":
        uploaded_file = st.file_uploader("Upload Sampel Daun", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    else:
        # st.camera_input: memungkinkan ambil foto dari kamera (jika device/browser mendukung)
        cam = st.camera_input("Ambil foto dari kamera")
        if cam is not None:
            image = Image.open(cam)

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
                    result = run_research_ensemble(models_dict, weights_dict, image)
                    st.session_state["diagnosis_result"] = result

    # ===== tampilkan hasil jika ada =====
    if st.session_state["diagnosis_result"] is not None:
        result = st.session_state["diagnosis_result"]

        st.subheader("✅ Hasil Diagnosis Ensemble")
        st.metric("Prediksi Akhir", result["final_label"], f"{result['final_conf']*100:.2f}%")

        # -------- DataFrame urut tetap (WAJIB untuk grafik berurutan) --------
        probs_df_ordered = pd.DataFrame({
            "Kelas": CLASS_NAMES,             # urutan sesuai index 0..3 (yang kamu minta)
            "Probabilitas": result["final_probs"]
        })

        # -------- DataFrame ranking (opsional untuk tabel “tertinggi dulu”) --------
        probs_df_rank = probs_df_ordered.sort_values("Probabilitas", ascending=False)

        st.write("**Probabilitas Ensemble (Weighted Average):**")
        st.dataframe(probs_df_rank, use_container_width=True)

        # ===== Grafik: PAKSA urutan sesuai CLASS_NAMES =====
        chart = alt.Chart(probs_df_ordered).mark_bar().encode(
            x=alt.X("Kelas:N", sort=CLASS_NAMES),  # penting: jangan pakai sort="-y"
            y=alt.Y("Probabilitas:Q"),
            tooltip=["Kelas", alt.Tooltip("Probabilitas:Q", format=".4f")]
        ).properties(height=280).configure_axisX(labelAngle=0)

        st.altair_chart(chart, use_container_width=True)

        # ===== Detail per model (raw probs) =====
        if len(result["model_names"]) > 0:
            st.subheader("🔬 Detail Probabilitas per Model")
            raw_df = pd.DataFrame(
                result["raw_matrix"],
                columns=CLASS_NAMES,
                index=result["model_names"]
            )
            st.dataframe(raw_df, use_container_width=True)

            st.caption(f"Total bobot efektif: **{result['effective_total_weight']:.2f}**")
            st.caption(f"Bobot yang dipakai: **{result['weights_used']}**")

# ============================================================
# TAB 2: CHATBOT (RAG dari SOP/PDF)
# ============================================================
with tab2:
    st.header("💬 Diskusi Pakar (RAG dari SOP/PDF)")

    if jumlah_pdf == 0 or vectorstore_db is None:
        st.warning("Belum ada PDF di folder `knowledge_base/`.")
    else:
        st.success("Knowledge base siap.")

    # tampilkan chat history
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
                answer = "Groq API Key belum di-set. Set via ENV `GROQ_API_KEY` atau `st.secrets`."
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
