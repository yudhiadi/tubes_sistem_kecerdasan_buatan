import streamlit as st
import os
import glob
import numpy as np
import pandas as pd
import altair as alt  # <--- [TAMBAHAN BARU] Untuk grafik yang tidak berubah urutannya
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- IMPORT FUNGSI PREPROCESSING BAWAAN ---
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_prep
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_prep

# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Sistem Pakar Jagung (Research Mode)", layout="wide")

if 'diagnosis_result' not in st.session_state:
    st.session_state['diagnosis_result'] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 2. LABEL & FUNGSI VISION
# ==========================================
# URUTAN LABEL (TETAP SESUAI INI)
CLASS_NAMES = [
    'Hawar Daun (Northern Leaf Blight)', # Index 0
    'Karat Daun (Common Rust)',          # Index 1
    'Bercak Daun (Gray Leaf Spot)',      # Index 2
    'Tanaman Sehat'                      # Index 3
]

@st.cache_resource
def load_all_models():
    models = {}
    model_files = {
        "EfficientNet": "model_jagung_efficientnet_vFinal.h5",
        "ResNet": "model_jagung_resnet_vFinal.h5",
        "MobileNet": "model_jagung_mobilenet_vFinal.h5"
    }
    loaded_count = 0
    for name, path in model_files.items():
        try:
            if os.path.exists(path):
                models[name] = load_model(path, compile=False)
                loaded_count += 1
        except Exception as e:
            print(f"Gagal load {name}: {e}")
    return models, loaded_count

def run_research_ensemble(models_dict, weights_dict, image):
    # 1. Resize gambar ke 224x224
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # 2. Ubah ke Array (Float32)
    img_array_base = np.asarray(image_resized).astype(np.float32)
    
    model_names = []
    raw_probs_matrix = []
    weighted_sum = np.zeros((1, 4))
    
    total_weight = sum(weights_dict.values())
    if total_weight == 0: total_weight = 1.0

    for name, model in models_dict.items():
        img_input = img_array_base.copy()
        
        # Preprocessing sesuai model
        if "MobileNet" in name:
            img_input = mobilenet_prep(img_input)
        elif "ResNet" in name:
            img_input = resnet_prep(img_input)
        elif "EfficientNet" in name:
            img_input = efficientnet_prep(img_input)
        
        img_input = np.expand_dims(img_input, axis=0)
        pred_prob = model.predict(img_input, verbose=0)[0]
        
        model_names.append(name)
        raw_probs_matrix.append(pred_prob)
        
        w = weights_dict.get(name, 0)
        weighted_sum += pred_prob * w

    final_ensemble_prob = weighted_sum / total_weight
    final_idx = np.argmax(final_ensemble_prob)
    
    return {
        "final_label": CLASS_NAMES[final_idx],
        "final_conf": final_ensemble_prob[0][final_idx],
        "final_probs": final_ensemble_prob[0],
        "model_names": model_names,
        "raw_matrix": np.array(raw_probs_matrix),
        "total_weight": total_weight
    }

# ==========================================
# 3. CHATBOT SETUP
# ==========================================
@st.cache_resource
def load_knowledge_base():
    SOP_FOLDER = "knowledge_base"
    if not os.path.exists(SOP_FOLDER): os.makedirs(SOP_FOLDER); return None, 0
    pdf_files = glob.glob(os.path.join(SOP_FOLDER, "*.pdf"))
    if not pdf_files: return None, 0
    all_documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path); docs = loader.load(); all_documents.extend(docs)
        except: pass
    if all_documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(all_documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embeddings)
        return vectorstore, len(pdf_files)
    return None, 0

vectorstore_db, jumlah_pdf = load_knowledge_base()

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🌽 Lab Riset Jagung")
    st.caption("Mode: Comparative Analysis & Ensemble")
    st.markdown("---")
    
    groq_api_key = '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'
    
    models_dict, count = load_all_models()
    
    st.markdown("### ⚖️ Konfigurasi Bobot")
    w_eff = st.number_input("Bobot EfficientNet", 0.0, 1.0, 0.5, 0.1)
    w_res = st.number_input("Bobot ResNet", 0.0, 1.0, 0.3, 0.1)
    w_mob = st.number_input("Bobot MobileNet", 0.0, 1.0, 0.2, 0.1)
    
    weights_dict = {"EfficientNet": w_eff, "ResNet": w_res, "MobileNet": w_mob}
    
    if count < 3:
        st.warning(f"⚠️ Hanya {count} model terdeteksi.")

# ==========================================
# 5. MAIN CONTENT
# ==========================================
tab1, tab2 = st.tabs(["📊 Analisis Citra & Data", "💬 Diskusi Pakar"])

with tab1:
    st.header("Analisis Citra (Untuk Keperluan Paper)")
    
    uploaded_file = st.file_uploader("Upload Sampel Daun", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        col_img, col_act = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Citra Input", use_container_width=True)
        with col_act:
            st.info("Klik tombol di bawah untuk menjalankan inferensi pada semua model.")
            run_btn = st.button("🚀 Jalankan Kalkulasi Multi-Model")

        if run_btn and count > 0:
            with st.spinner("Menghitung matriks probabilitas..."):
                res = run_research_ensemble(models_dict, weights_dict, image)
                st.session_state['diagnosis_result'] = res['final_label']
                
                # --- HASIL AKHIR ---
                st.divider()
                st.subheader("1. Hasil Akhir (Weighted Ensemble)")
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"**Prediksi Final:** {res['final_label']}")
                with c2:
                    st.metric("Confidence Score", f"{res['final_conf']*100:.2f}%")

                # --- MATRIKS DATA ---
                st.subheader("2. Matriks Probabilitas (Data Mentah)")
                df_raw = pd.DataFrame(res['raw_matrix'], columns=CLASS_NAMES, index=res['model_names'])
                df_raw.loc['**ENSEMBLE RESULT**'] = res['final_probs']
                st.dataframe(df_raw.style.format("{:.2%}"), use_container_width=True)
                
                # --- [BAGIAN INI YANG DIPERBAIKI UNTUK VISUALISASI] ---
                st.subheader("3. Visualisasi Perbandingan")
                
                # Kita siapkan data khusus untuk chart (exclude baris Ensemble)
                chart_df = df_raw.iloc[:-1].T.reset_index()
                chart_df.columns = ['Penyakit'] + res['model_names']
                
                # Ubah format data menjadi 'Long Format'
                chart_df_long = chart_df.melt('Penyakit', var_name='Model', value_name='Probabilitas')
                
                # Buat Chart Grouped Bar
                chart = alt.Chart(chart_df_long).mark_bar().encode(
                    x=alt.X('Penyakit', sort=CLASS_NAMES, title='Kelas Penyakit', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Probabilitas', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])), # Skala dikunci 0-100%
                    color='Model',
                    xOffset='Model', # <--- INI KUNCINYA: Supaya batang berjejer ke samping (Grouped)
                    tooltip=['Penyakit', 'Model', alt.Tooltip('Probabilitas', format='.2%')]
                ).properties(
                    height=400 
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)

# --- TAB 2: CHATBOT RAG ---
with tab2:
    st.header("Diskusi Pakar (RAG)")
    if st.session_state['diagnosis_result']:
        st.info(f"Topik Diskusi: **{st.session_state['diagnosis_result']}**")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Tanya tentang hasil analisis di atas..."):
        if not groq_api_key:
            st.error("Masukkan API Key Groq di Sidebar dulu!")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)

            with st.chat_message("assistant"):
                if not vectorstore_db:
                    context = "Tidak ada dokumen PDF ditemukan."
                else:
                    retriever = vectorstore_db.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.invoke(user_input)
                    context = "\n".join([d.page_content for d in docs])

                prompt = f"""
                Kamu adalah asisten peneliti pertanian.
                Hasil deteksi sistem: {st.session_state['diagnosis_result']}
                Referensi: {context}
                User: {user_input}
                Jawab ilmiah namun praktis. Gunakan Bahasa Indonesia.
                """
                try:
                    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
                    resp = llm.invoke(prompt)
                    st.markdown(resp.content)
                    st.session_state.messages.append({"role": "assistant", "content": resp.content})
                except Exception as e:
                    st.error(f"Error: {e}")