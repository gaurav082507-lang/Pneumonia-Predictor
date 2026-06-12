import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# 1. Page Layout Configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="centered"
)

# 2. Native Hugging Face Cloud Downloader
@st.cache_resource
def load_cloud_model():
    model_path = "pneumonia_model.h5"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading heavy AI model weights from Hugging Face Hub... Please wait."):
            # ⚠️ REPLACE 'your-username' WITH YOUR ACTUAL HUGGING FACE USERNAME ⚠️
            # ⚠️ REPLACE 'pneumonia-model' WITH YOUR ACTUAL HF REPOSITORY NAME ⚠️
            model_path = hf_hub_download(
                repo_id="MLCoder90/Pneumonia_Model", 
                filename="pneumonia_model.h5"
            )
            
    return tf.keras.models.load_model(model_path)

try:
    model = load_cloud_model()
except Exception as e:
    st.error(f"Failed to load model from Hugging Face. Error details: {e}")
    st.stop()

# 3. Clean Dashboard UI Header
st.title("🫁 Chest X-Ray Pneumonia Detector")
st.write("Upload a patient's frontal chest X-ray image (JPEG format) for rapid deep learning feature analysis.")
st.markdown("---")

# 4. Image Uploader Component
uploaded_file = st.file_uploader("Upload Chest X-Ray Image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Chest X-Ray Scan", use_container_width=True)
    
    st.write("🔄 Processing matrix dimensions and computing model inference...")
    
    # 5. Image Preprocessing 
    img_array = np.array(image)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_resized = tf.image.resize(img_tensor, size=[224, 224])
    img_normalized = img_resized / 255.0
    img_batch = tf.expand_dims(img_normalized, axis=0)
    
    # 6. Model Prediction Computation
    prediction_raw = model.predict(img_batch)[0][0]
    
    # 7. Dynamic UI Triage Banner Outputs
    if 0.45 <= prediction_raw <= 0.55:
        st.warning("⚠️ **Result: Inconclusive Scan / High Uncertainty**")
        st.info("The structural features in this scan do not clearly align with either baseline. A manual clinical review is highly recommended.")
    else:
        if prediction_raw > 0.5:
            confidence = prediction_raw * 100
            st.error("🚨 **Result: PNEUMONIA DETECTED**")
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
            st.markdown("> **Clinical Note:** Increased density, consolidations, or fluid patterns detected in lung fields.")
        else:
            confidence = (1 - prediction_raw) * 100
            st.success("🟢 **Result: NORMAL / HEALTHY**")
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
            st.markdown("> **Clinical Note:** Clear lung fields without distinct signatures of bacterial or viral fluid infiltration.")

# 8. Clean Branded Footer with Dynamic Disclaimer
st.markdown("---")
st.caption("🚀 **Built by Gaurav Gupta (BIT Mesra AIML)**")
st.caption("Disclaimer: This deep learning model is an educational tool built for research verification. It is not an FDA-approved diagnostic instrument.")
