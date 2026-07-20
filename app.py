import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import InferenceClient
from PIL import Image

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

@st.cache_resource
def load_models():
    corn = load_model("corn_disease_model.h5")
    cocoa = load_model("cocoa_disease_model.h5")
    return corn, cocoa

corn_model, cocoa_model = load_models()

corn_classes = ['common_rust', 'gray_leaf_spot', 'healthy', 'northern_leaf_blight']
cocoa_classes = ['black_pod_rot', 'healthy', 'pod_borer']

knowledge_base = {
    ("corn", "healthy"): "No disease detected. Continue routine monitoring and standard field practices.",
    ("corn", "common_rust"): "Common rust causes reddish-brown pustules on leaves. Apply a fungicide containing azoxystrobin or propiconazole at early signs. Rotate crops and use rust-resistant maize varieties in future planting.",
    ("corn", "gray_leaf_spot"): "Gray leaf spot causes rectangular gray-brown lesions on leaves. Apply a strobilurin-based fungicide. Practice crop rotation and avoid dense planting to improve air circulation.",
    ("corn", "northern_leaf_blight"): "Northern leaf blight causes long, cigar-shaped gray-green lesions. Apply fungicide at first sign of infection. Remove and destroy infected crop debris after harvest, and use resistant hybrids where possible.",
    ("cocoa", "healthy"): "No disease detected. Continue routine monitoring and standard field practices.",
    ("cocoa", "black_pod_rot"): "Black pod rot causes dark, water-soaked lesions that spread across the pod. Remove and destroy infected pods immediately to prevent spread. Improve
