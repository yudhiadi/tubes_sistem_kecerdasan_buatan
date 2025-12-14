import os
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image, ImageOps

# --- IMPOR KRUSIAL UNTUK TF 2.19.0 / KERAS 3 ---
import tensorflow as tf
from tensorflow.keras.models import load_model 

# Preprocess sesuai backbone
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.densenet import preprocess_input as densenet_prep

# LLM (Groq)
from groq import Groq


# =========================
# KONFIG STREAMLIT & PATH
# =========================
st.set_page_config(page_title="Lab Riset Jagung - Ensemble + Chat Ahli", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LABEL KELAS & METADATA
# =========================
CLASS_NAMES = [
    "Hawar Daun (Northern Leaf Blight)", 
    "Karat Daun (Common Rust)",
    "Bercak Daun (Gray Leaf Spot)",
    "Tanaman Sehat"
]
N_CLASS = len(CLASS_NAMES)

# =========================
# MODEL FILES & PREPROCESSORS
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

# Bobot default
AUTO_WEIGHTS = {
    "MobileNetV3": 0.333,
    "EfficientNet": 0.332,
    "DenseNet": 0.335,
}


# =========================
# Helper: cek file & LFS pointer
# =========================
def _is_lfs_pointer(head_bytes: bytes) -> bool:
    txt = head_bytes.decode("utf-8", errors="ignore")
    return "git-lfs.github.com/spec/v1" in txt

def inspect_model_file(path: str) -> dict:
    info = {"exists": os.path.exists(path), "size_mb": None, "is_lfs_pointer": False}
    if not info["exists"]:
        return info
    try:
        info["size_mb"] = round(os.path.getsize(path) / (1024 * 1024), 3)
        with open(path, "rb") as f:
            head = f.read(200)
        info["is_lfs_pointer"] = _is_lfs_pointer(head)
    except Exception:
        pass
    return info


# =========================
# LOAD MODELS (cache) + diagnostics
# =========================
@st.cache_resource
def load_all_models():
    models = {}
    diagnostics = {}

    for name, filename in MODEL_FILES.items():
        full_path = os.path.join(BASE_DIR, filename)
        finfo = inspect_model_file(full_path)

        diagnostics[name] = {
            "file": filename,
            "path": full_path,
            "inspect": finfo,
            "load_ok": False,
            "error": None,
            "traceback": None,
        }

        if not finfo["exists"]:
            diagnostics[name]["error"] = "File tidak ditemukan di server."
            continue
        if finfo["is_lfs_pointer"]:
            diagnostics[name]["error"] = "Terdeteksi Git LFS pointer (bukan file model asli)."
            continue

        try:
            models[name] = load_model(full_path, compile=False) 
            diagnostics[name]["load_ok"] = True
        except Exception as e:
            # Recovery
            try:
                models[name] = load_model(
                    full_path, 
                    compile=False, 
                    custom_objects={"relu6": tf.nn.relu6, "Functional": tf.keras.models.Functional}
                )
                diagnostics[name]["load_ok"] = True
            except Exception as e2:
                diagnostics[name]["error"] = f"Gagal Deserialisasi Model. Pastikan requirements.txt menggunakan: tensorflow==2.19.0. Detail: {str(e2)}"
                diagnostics[name]["traceback"] = traceback.format_exc()

    return models, diagnostics


# =========================
# ENSEMBLE INFERENCE
# =========================
def preprocess_for_model(image_pil: Image.Image, model_name: str, size=(224, 224)) -> np.ndarray:
    img = ImageOps.fit(image_pil, size, Image.Resampling.LANCZOS).convert("RGB")
    arr = np.asarray(img).astype(np.float32) 
    arr = PREPROCESS_MAP[model_name](arr) 
    arr = np.expand_dims(arr, axis=0) 
    return arr

def run_ensemble(models_dict: dict, weights: dict, image_pil: Image.Image) -> dict:
    weighted_sum = np.zeros((N_CLASS,), dtype=np.float32)
    total_w = 0.0

    per_model_probs = {}
    used_models = []

    for name, model in models_dict.items():
        if name not in weights:
            continue
        w = float(weights[name])
        if w <= 0:
            continue

        x = preprocess_for_model(image_pil, name)
        probs = model.predict(x, verbose=0)[0] 
        probs = np.asarray(probs, dtype=np.float32)

        per_model_probs[name] = probs
        used_models.append(name)

        weighted_sum += probs * w
        total_w += w

    if total_w <= 0:
        total_w = 1.0

    final_probs = weighted_sum / total_w
    final_idx = int(np.argmax(final_probs))

    return {
        "used_models": used_models,
        "per_model_probs": per_model_probs,
        "final_probs": final_probs,
        "final_label": CLASS_NAMES[final_idx],
        "final_conf": float(final_probs[final_idx]),
        "total_weight": float(total_w),
    }


# =========================
# KNOWLEDGE BASE
# =========================
KB_DOCS = [
    {
        "title": "Hawar Daun (Northern Leaf Blight)",
        "text": (
            "Gejala: bercak memanjang seperti cerutu/oval abu-abu kecoklatan, meluas pada daun tua.\n"
            "Penyebab: jamur (umumnya Exserohilum turcicum).\n"
            "Penanganan: gunakan varietas toleran, rotasi tanaman, buang sisa tanaman terinfeksi, "
            "perbaiki sirkulasi udara, fungisida sesuai anjuran (jika serangan berat).\n"
            "Pencegahan: jarak tanam tepat, sanitasi lahan, pemupukan seimbang (hindari N berlebih)."
        )
    },
    {
        "title": "Karat Daun (Common Rust)",
        "text": (
            "Gejala: pustula/bintik menonjol berwarna coklat kemerahan seperti karat, bisa menyebar cepat.\n"
            "Penyebab: jamur Puccinia sorghi.\n"
            "Penanganan: varietas tahan, monitoring awal, fungisida bila ambang serangan tinggi.\n"
            "Pencegahan: kurangi kelembapan berlebih, sanitasi, tanam serempak bila memungkinkan."
        )
    },
    {
        "title": "Bercak Daun (Gray Leaf Spot)",
        "text": (
            "Gejala: bercak persegi panjang abu-abu keperakan mengikuti tulang daun, sering muncul pada kondisi lembap.\n"
            "Penyebab: jamur Cercospora zeae-maydis.\n"
            "Penanganan: residu tanaman dikelola (dibajak/kompos), rotasi, varietas toleran, fungisida sesuai kebutuhan.\n"
            "Pencegahan: sirkulasi udara baik, tidak terlalu rapat, hindari daun lama lembap terlalu lama."
        )
    },
    {
        "title": "Tanaman Sehat - Praktik Umum",
        "text": (
            "Praktik baik: benih bermutu, pemupukan berimbang, pengairan cukup, kontrol gulma, "
            "monitor hama-penyakit rutin, dan sanitasi lahan.\n"
            "Jika ada gejala mirip penyakit: dokumentasikan (foto), cek sebaran, cek riwayat cuaca & kelembapan, "
            "lakukan tindakan bertahap (budidaya dulu sebelum kimia)."
        )
    },
]

def simple_retrieve(query: str, k: int = 2):
    q = (query or "").lower()
    scored = []
    for d in KB_DOCS:
        text = (d["title"] + " " + d["text"]).lower()
        score = sum(1 for w in q.split() if w in text)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:k] if s > 0]


# ==================================
# ✅ SESSION STATE (DIPINDAHKAN KE ATAS)
# ==================================
# Memastikan state terinisialisasi sebelum kode UI mencoba membacanya
if "diagnosis" not in st.session_state:
    # Nilai default yang lengkap untuk menghindari error .get("final_label")
    st.session_state["diagnosis"] = {"final_label": "Belum Didiagnosis"} 

if "chat" not in st.session_state:
    st.session_state["chat"] = [
        {"role": "assistant", "content": "Halo! Upload foto daun jagung di tab Analisis, atau tanya saya tentang penyakit & penanganan jagung."}
    ]


# =========================
# LOAD MODELS (sekali)
# =========================
models_dict, model_diag = load_all_models()
loaded_models = list(models_dict.keys())


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Ensemble 3 model + Chat Ahli Jagung")
    st.markdown("---")

    st.markdown("### 🧠 Status Model")
    st.caption(f"TensorFlow: {tf.__version__}")
    
    # Peringatan Versi
    if tf.__version__ != '2.19.0':
        st.warning(f"PERHATIAN: Versi TF di sini ({tf.__version__}) TIDAK SAMA dengan versi training (2.19.0). Gagal load model mungkin terjadi!")
    
    for m in MODEL_FILES.keys():
        d = model_diag[m]
        if d["load_ok"]:
            st.write(f"✅ {m} loaded")
        else:
            st.write(f"❌ {m} gagal")
            with st.expander(f"Detail error: {m}", expanded=False):
                st.write("Path:", d["path"])
                st.write("Inspect:", d["inspect"])
                st.error(d["error"]) 
                if d["traceback"]:
                    st.code(d["traceback"])

    st.markdown("---")
    st.markdown("### ⚖️ Bobot Ensemble")
    use_auto = st.checkbox("Gunakan bobot default", value=True)

    # Bobot (tidak diubah)
    if use_auto:
        weights = dict(AUTO_WEIGHTS)
        st.info(
            f"MobileNetV3={weights['MobileNetV3']:.3f}, "
            f"EfficientNet={weights['EfficientNet']:.3f}, "
            f"DenseNet={weights['DenseNet']:.3f}"
        )
    else:
        w_m = st.number_input("Bobot MobileNetV3", 0.0, 1.0, float(AUTO_WEIGHTS["MobileNetV3"]), 0.01)
        w_e = st.number_input("Bobot EfficientNet", 0.0, 1.0, float(AUTO_WEIGHTS["EfficientNet"]), 0.01)
        w_d = st.number_input("Bobot DenseNet", 0.0, 1.0, float(AUTO_WEIGHTS["DenseNet"]), 0.01)
        weights = {"MobileNetV3": w_m, "EfficientNet": w_e, "DenseNet": w_d}

        if st.checkbox("Normalisasi (sum=1)", value=True):
            s = sum(weights.values())
            if s > 0:
                weights = {k: v / s for k, v in weights.items()}

    st.markdown("---")
    # Groq API Key
    groq_api_key = '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'
    if not groq_api_key:
        st.error("GROQ_API_KEY belum terkonfigurasi. Chat Ahli mungkin tidak berfungsi.")
        groq_api_key = st.text_input("Masukkan Groq API Key (opsional)", type="password")


# =========================
# MAIN UI
# =========================
tab1, tab2 = st.tabs(["📊 Analisis Gambar (Ensemble)", "💬 Chat Ahli Jagung (LLM + KB)"])


# ---------- TAB 1 (Analisis Gambar) ----------
with tab1:
    st.header("📊 Analisis Gambar Daun Jagung (Ensemble 3 Model)")
    st.caption("Upload foto daun → sistem menghitung probabilitas tiap kelas dan hasil ensemble.")

    uploaded = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    image = Image.open(uploaded).convert("RGB") if uploaded else None

    if image:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(image, caption="Citra Input", use_container_width=True)

        with col2:
            if len(loaded_models) == 0:
                st.error("Tidak ada model yang berhasil dimuat. Cek sidebar error detail.")
            else:
                st.write("Model aktif:", ", ".join(loaded_models))
                
                # JANGAN inisialisasi st.session_state["diagnosis"] di sini lagi! 
                # Sudah dilakukan di awal script.
                
                if st.button("🔎 Jalankan Prediksi", use_container_width=True):
                    with st.spinner("Memproses..."):
                        result = run_ensemble(models_dict, weights, image)
                        st.session_state["diagnosis"] = result
                    # Gunakan st.rerun() di Streamlit modern
                    st.rerun() 

    # Tampilkan Hasil Ensemble jika ada diagnosis
    if st.session_state["diagnosis"]["final_label"] != "Belum Didiagnosis":
        res = st.session_state["diagnosis"]

        st.subheader("✅ Hasil Ensemble")
        st.metric("Prediksi Akhir", res["final_label"], f"{res['final_conf']*100:.2f}%")
        st.caption(f"Model terpakai: {', '.join(res['used_models'])} | Total bobot efektif: {res['total_weight']:.3f}")

        probs_df = pd.DataFrame({"Kelas": CLASS_NAMES, "Probabilitas": res["final_probs"]})
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

        # detail per model
        if res["per_model_probs"]:
            st.subheader("🔬 Probabilitas per Model")
            raw_df = pd.DataFrame({m: res["per_model_probs"][m] for m in res["per_model_probs"]}, index=CLASS_NAMES)
            st.dataframe(raw_df.T, use_container_width=True)


# ---------- TAB 2 (Chat Ahli) ----------
with tab2:
    st.header("💬 Chat Ahli Tanaman Jagung")

    # AKSES STATE YANG LEBIH AMAN (Sudah dijamin ada karena inisialisasi di atas)
    last_diag = st.session_state["diagnosis"]["final_label"] 
    
    if last_diag != "Belum Didiagnosis":
        st.info(f"Diagnosis terakhir dari gambar: **{last_diag}** (kamu bisa tanya penanganannya).")
    else:
        st.info("Belum ada diagnosis gambar. Kamu tetap bisa tanya tentang budidaya/penyakit jagung.")

    # tampilkan chat history
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Tulis pertanyaan tentang jagung…")
    if user_q:
        st.session_state["chat"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # retrieval KB sederhana
        hits = simple_retrieve(user_q, k=2)
        if hits:
            kb_context = "\n\n".join([f"- {h['title']}:\n{h['text']}" for h in hits])
        else:
            kb_context = "\n".join([f"- {d['title']}:\n{d['text']}" for d in KB_DOCS])

        diag_context = f"Diagnosis visual terakhir: {last_diag}." if last_diag != "Belum Didiagnosis" else "Tidak ada diagnosis visual."

        prompt = f"""
Anda adalah ahli tanaman jagung (penyakit daun, budidaya, pencegahan, dan penanganan).
Jawab dalam bahasa Indonesia yang lugas, terstruktur, dan aman.

Konteks:
- {diag_context}

Knowledge base (rujukan utama bila relevan):
{kb_context}

Aturan jawaban:
- Berikan langkah penanganan bertahap: identifikasi → tindakan budidaya → opsi kimia (bila perlu) → pencegahan.
- Jika perlu klarifikasi, ajukan 2–3 pertanyaan singkat.
- Jangan mengarang dosis pestisida spesifik; sarankan ikuti label produk & rekomendasi penyuluh setempat.

Pertanyaan pengguna:
{user_q}

Jawaban ahli:
""".strip()

        if not groq_api_key:
            answer = (
                "Saya belum bisa akses LLM karena Groq API Key belum di-set.\n\n"
                "Berikut referensi knowledge base yang relevan:\n"
                f"{kb_context}"
            )
        else:
            try:
                client = Groq(api_key=groq_api_key)
                with st.spinner("Ahli AI sedang menyiapkan jawaban..."):
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Anda adalah ahli tanaman jagung yang membantu diagnosis dan penanganan penyakit secara aman dan praktis."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                    )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Error koneksi Groq: {e}\n\n(Kamu masih bisa pakai knowledge base lokal:)\n{kb_context}"

        st.session_state["chat"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)