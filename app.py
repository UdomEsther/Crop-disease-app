import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import InferenceClient
from PIL import Image

st.set_page_config(page_title="Crop Disease Detector", page_icon="🌱")

# ---------- Load models (once, cached) ----------
@st.cache_resource
def load_models():
    corn = load_model("corn_disease_model.h5")
    cocoa = load_model("cocoa_disease_model.h5")
    return corn, cocoa

corn_model, cocoa_model = load_models()

corn_classes = ['common_rust', 'gray_leaf_spot', 'healthy', 'northern_leaf_blight']
cocoa_classes = ['black_pod_rot', 'healthy', 'pod_borer']

# ---------- Knowledge base ----------
knowledge_base = {
    ("corn", "healthy"): "No disease detected. Continue routine monitoring and standard field practices.",
    ("corn", "common_rust"): "Common rust causes reddish-brown pustules on leaves. Apply a fungicide containing azoxystrobin or propiconazole at early signs. Rotate crops and use rust-resistant maize varieties in future planting.",
    ("corn", "gray_leaf_spot"): "Gray leaf spot causes rectangular gray-brown lesions on leaves. Apply a strobilurin-based fungicide. Practice crop rotation and avoid dense planting to improve air circulation.",
    ("corn", "northern_leaf_blight"): "Northern leaf blight causes long, cigar-shaped gray-green lesions. Apply fungicide at first sign of infection. Remove and destroy infected crop debris after harvest, and use resistant hybrids where possible.",
    ("cocoa", "healthy"): "No disease detected. Continue routine monitoring and standard field practices.",
    ("cocoa", "black_pod_rot"): "Black pod rot causes dark, water-soaked lesions that spread across the pod. Remove and destroy infected pods immediately to prevent spread. Improve field drainage and apply a copper-based fungicide.",
    ("cocoa", "pod_borer"): "Pod borer infestation shows as small holes and internal tunnelling in pods. Remove and destroy infested pods. Use pheromone traps to monitor adult moths, and apply approved insecticide if infestation is severe.",
}

# ---------- Hugging Face client (token from Streamlit secrets, never hardcoded) ----------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
client = InferenceClient(provider="publicai", api_key=HF_TOKEN) if HF_TOKEN else None


def get_recommendation(crop, disease, confidence):
    base_fact = knowledge_base.get((crop, disease), "No information available for this diagnosis.")

    # If no HF token is configured, just show the technical note directly
    if client is None:
        return base_fact

    prompt = f"""You are helping a farmer understand a crop diagnosis. Rewrite the following technical note as a short, warm, easy-to-understand message for a farmer with no scientific background. Keep it under 80 words. Do not add any information beyond what is given.

Crop: {crop}
Diagnosis: {disease.replace('_', ' ')}
Confidence: {confidence:.0%}
Technical note: {base_fact}"""

    try:
        response = client.chat.completions.create(
            model="swiss-ai/Apertus-70B-Instruct-2509",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # If the AI rewrite fails for any reason, fall back to the plain technical note
        return base_fact


def diagnose_crop(image: Image.Image, crop_type: str):
    """
    image: a PIL Image (from camera or upload)
    crop_type: 'corn' or 'cocoa'
    """
    img = image.convert("RGB").resize((224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    if crop_type == "corn":
        model = corn_model
        classes = corn_classes
    elif crop_type == "cocoa":
        model = cocoa_model
        classes = cocoa_classes
    else:
        raise ValueError("crop_type must be 'corn' or 'cocoa'")

    prediction = model.predict(x, verbose=0)
    predicted_index = np.argmax(prediction[0])
    disease = classes[predicted_index]
    confidence = float(prediction[0][predicted_index])

    recommendation = get_recommendation(crop_type, disease, confidence)

    return {
        "crop": crop_type,
        "diagnosis": disease,
        "confidence": round(confidence, 4),
        "recommendation": recommendation,
    }


# ---------- UI ----------
st.title("🌱 Crop Disease Detector")
st.write("Take a photo or upload one of your corn or cocoa crop to check for disease.")

crop_type = st.radio("Which crop is this?", ["corn", "cocoa"], horizontal=True)

st.markdown("**Step 1: Get a photo**")
camera_photo = st.camera_input("📷 Take a photo")
uploaded_photo = st.file_uploader("...or upload one from your gallery", type=["jpg", "jpeg", "png"])

image_file = camera_photo if camera_photo is not None else uploaded_photo

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Your photo", use_container_width=True)

    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing your crop..."):
            result = diagnose_crop(image, crop_type)

        st.markdown("---")
        if result["diagnosis"] == "healthy":
            st.success(f"✅ Diagnosis: **Healthy** ({result['confidence']:.0%} confidence)")
        else:
            st.warning(
                f"⚠️ Diagnosis: **{result['diagnosis'].replace('_', ' ').title()}** "
                f"({result['confidence']:.0%} confidence)"
            )
        st.markdown("### 🌾 What this means for you")
        st.write(result["recommendation"])
else:
    st.info("Take or upload a photo above to get started.")
