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
st.set_page_config(page_title="Corn Research Lab - Ensemble + Expert Chat", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# LABEL KELAS & METADATA
# =========================
CLASS_NAMES = [
    "Northern Leaf Blight",
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy Plant"
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
            diagnostics[name]["error"] = "File not found on server."
            continue
        if finfo["is_lfs_pointer"]:
            diagnostics[name]["error"] = "Detected Git LFS pointer (not the actual model file)."
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
        "title": "Northern Leaf Blight",
        "text": (
            "Symptoms: elongated cigar-shaped or oval grayish-brown lesions, spreading on older leaves.\n"
            "Cause: fungus (commonly Exserohilum turcicum).\n"
            "Management: use tolerant varieties, crop rotation, remove infected plant debris, "
            "improve air circulation, fungicide as recommended (if severe infection).\n"
            "Prevention: proper plant spacing, field sanitation, balanced fertilization (avoid excess nitrogen)."
        )
    },
    {
        "title": "Common Rust",
        "text": (
            "Symptoms: raised reddish-brown pustules/spots resembling rust, can spread rapidly.\n"
            "Cause: fungus Puccinia sorghi.\n"
            "Management: resistant varieties, early monitoring, fungicide if infection threshold is high.\n"
            "Prevention: reduce excess humidity, sanitation, synchronized planting if possible."
        )
    },
    {
        "title": "Gray Leaf Spot",
        "text": (
            "Symptoms: rectangular grayish-silver spots following leaf veins, often appear in humid conditions.\n"
            "Cause: fungus Cercospora zeae-maydis.\n"
            "Management: manage crop residues (plow/compost), rotation, tolerant varieties, fungicide as needed.\n"
            "Prevention: good air circulation, avoid overcrowding, prevent old leaves from staying wet too long."
        )
    },
    {
        "title": "Healthy Plant - General Practices",
        "text": (
            "Good practices: quality seeds, balanced fertilization, adequate irrigation, weed control, "
            "routine pest-disease monitoring, and field sanitation.\n"
            "If symptoms similar to disease appear: document (photo), check distribution, review weather & humidity history, "
            "take stepwise actions (cultural first before chemical)."
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
        {"role": "assistant", "content": "Hello! Upload a corn leaf photo in the Analysis tab, or ask me about corn diseases and management."}
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
    # if tf.__version__ != '2.19.0':
    #     st.warning(f"PERHATIAN: Versi TF di sini ({tf.__version__}) TIDAK SAMA dengan versi training (2.19.0). Gagal load model mungkin terjadi!")
    
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
        w_m = st.number_input("MobileNetV3 Weight", 0.0, 1.0, float(AUTO_WEIGHTS["MobileNetV3"]), 0.01)
        w_e = st.number_input("EfficientNet Weight", 0.0, 1.0, float(AUTO_WEIGHTS["EfficientNet"]), 0.01)
        w_d = st.number_input("DenseNet Weight", 0.0, 1.0, float(AUTO_WEIGHTS["DenseNet"]), 0.01)
        weights = {"MobileNetV3": w_m, "EfficientNet": w_e, "DenseNet": w_d}

        if st.checkbox("Normalisasi (sum=1)", value=True):
            s = sum(weights.values())
            if s > 0:
                weights = {k: v / s for k, v in weights.items()}

    st.markdown("---")
    # Groq API Key
    groq_api_key = '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'
    if not groq_api_key:
        st.error("GROQ_API_KEY is not configured. Expert Chat may not work.")
        groq_api_key = st.text_input("Enter Groq API Key (optional)", type="password")


# ---------- TAB 1 (Analisis Gambar) ----------
with tab1:
    st.header("📊 Analisis Gambar Daun Jagung (Ensemble 3 Model)")
    st.caption("Upload foto daun → sistem menghitung probabilitas tiap kelas dan hasil ensemble.")

    uploaded = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    image = Image.open(uploaded).convert("RGB") if uploaded else None

    if image:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with col2:
            if len(loaded_models) == 0:
                st.error("No models loaded successfully. Check sidebar for error details.")
            else:
                st.write("Active models:", ", ".join(loaded_models))
                
                # DO NOT reinitialize st.session_state["diagnosis"] here! 
                # Already done at the top of the script.
                
                if st.button("🔎 Run Prediction", use_container_width=True):
                    with st.spinner("Processing..."):
                        result = run_ensemble(models_dict, weights, image)
                        st.session_state["diagnosis"] = result
                    # Use st.rerun() in modern Streamlit
                    st.rerun()

    # Show Ensemble Result if diagnosis exists
    if st.session_state["diagnosis"]["final_label"] != "Belum Didiagnosis":
        res = st.session_state["diagnosis"]

        st.subheader("✅ Ensemble Result")
        st.metric("Final Prediction", res["final_label"], f"{res['final_conf']*100:.2f}%")
        st.caption(f"Models used: {', '.join(res['used_models'])} | Total effective weight: {res['total_weight']:.3f}")

        probs_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": res["final_probs"]})
        st.dataframe(probs_df, use_container_width=True)

        chart = (
            alt.Chart(probs_df)
            .mark_bar()
            .encode(
                x=alt.X("Class:N", sort=CLASS_NAMES),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Class", alt.Tooltip("Probability:Q", format=".4f")]
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

        # per-model details
        if res["per_model_probs"]:
            st.subheader("🔬 Probability per Model")
            raw_df = pd.DataFrame({m: res["per_model_probs"][m] for m in res["per_model_probs"]}, index=CLASS_NAMES)
            st.dataframe(raw_df.T, use_container_width=True)


# ---------- TAB 2 (Expert Chat) ----------
with tab2:
    st.header("💬 Corn Expert Chat")

    # Safe state access (already initialized above)
    last_diag = st.session_state["diagnosis"]["final_label"] 
    
    if last_diag != "Belum Didiagnosis":
        st.info(f"Last image diagnosis: **{last_diag}** (you can ask about its management).")
    else:
        st.info("No image diagnosis yet. You can still ask about corn cultivation/diseases.")

    # show chat history
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question about corn…")
    if user_q:
        st.session_state["chat"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # simple KB retrieval
        hits = simple_retrieve(user_q, k=2)
        if hits:
            kb_context = "\n\n".join([f"- {h['title']}:\n{h['text']}" for h in hits])
        else:
            kb_context = "\n".join([f"- {d['title']}:\n{d['text']}" for d in KB_DOCS])

        diag_context = f"Last visual diagnosis: {last_diag}." if last_diag != "Belum Didiagnosis" else "No visual diagnosis."

        prompt = f"""
You are a corn plant expert (leaf diseases, cultivation, prevention, and management).
Answer in clear, structured, and safe English.

Context:
- {diag_context}

Knowledge base (main reference if relevant):
{kb_context}

Answer rules:
- Provide stepwise management: identification → cultivation actions → chemical options (if needed) → prevention.
- If clarification is needed, ask 2–3 short questions.
- Do not invent specific pesticide dosages; advise to follow product label & local extension recommendations.

User question:
{user_q}

Expert answer:
""".strip()

        if not groq_api_key:
            answer = (
                "Cannot access LLM because Groq API Key is not set.\n\n"
                "Here are relevant knowledge base references:\n"
                f"{kb_context}"
            )
        else:
            try:
                client = Groq(api_key=groq_api_key)
                with st.spinner("AI Expert is preparing the answer..."):
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a corn plant expert who helps diagnose and manage diseases safely and practically."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                    )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"Groq connection error: {e}\n\n(You can still use the local knowledge base:)\n{kb_context}"

        st.session_state["chat"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)