import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Damage Detector",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

/* Main background */
.stApp {
    background: #0f0f13;
    color: #e8e6e1;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #16161d !important;
    border-right: 1px solid #2a2a35;
}

/* Upload box */
.stFileUploader > label {
    font-family: 'Syne', sans-serif;
    color: #e8e6e1 !important;
    font-weight: 700;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #3d3d52 !important;
    border-radius: 12px !important;
    padding: 20px;
    background: #16161d;
    transition: border-color 0.3s;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #e85d04 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #e85d04, #f48c06);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 16px;
    padding: 12px 32px;
    width: 100%;
    transition: all 0.3s;
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(232, 93, 4, 0.4);
}

/* Metric cards */
.metric-card {
    background: #16161d;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}

/* Result cards */
.result-damaged {
    background: linear-gradient(135deg, #1a0800, #2d1200);
    border: 2px solid #e85d04;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
}

.result-clean {
    background: linear-gradient(135deg, #001a0d, #002d1a);
    border: 2px solid #2dc653;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.result-subtitle {
    font-size: 0.95rem;
    opacity: 0.7;
    margin: 0;
}

.confidence-bar-bg {
    background: #2a2a35;
    border-radius: 999px;
    height: 8px;
    margin: 16px 0 4px;
    overflow: hidden;
}

.confidence-bar-fill-dmg {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #e85d04, #f48c06);
    transition: width 1s ease;
}

.confidence-bar-fill-ok {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #2dc653, #80ed99);
    transition: width 1s ease;
}

/* Info boxes */
.info-box {
    background: #1c1c26;
    border-left: 3px solid #e85d04;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #b0aead;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #2a2a35;
    margin: 24px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained Keras model."""
    model_path = "model/car_damage_model.h5"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, image: np.ndarray):
    """
    Returns (label_index, probabilities).
    label 0 = Damaged, label 1 = Undamaged
    """
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.reshape(img, [1, 128, 128, 3])
    probs = model.predict(img, verbose=0)[0]          # shape (2,)
    label = int(np.argmax(probs))
    return label, probs


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Car Damage Detector")
    st.markdown("---")
    st.markdown("""
**How it works**

1. Upload a photo of a car
2. The CNN model analyses the image
3. Get an instant damage assessment

---
**Model Architecture**

| Layer | Details |
|---|---|
| Input | 128×128 RGB |
| Conv2D ×3 | 32 → 64 → 128 filters |
| Dense | 128 → 64 → 32 units |
| Output | 2-class Softmax |

---
**Labels**
- 🔴 **Damaged** — visible damage
- 🟢 **Undamaged** — no visible damage

---
""")
    st.markdown("<div class='info-box'>Upload a clear, well-lit side/front/rear shot for best accuracy.</div>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Car Damage Detection")
st.markdown("##### AI-powered vehicle inspection using a Convolutional Neural Network")
st.markdown("---")

model = load_model()
if model is None:
    st.warning(
        "⚠️ **Model not found.**\n\n"
        "Place your trained model at `model/car_damage_model.h5` "
        "and restart the app. Run `train.py` to generate it.",
        icon="⚠️",
    )

uploaded = st.file_uploader(
    "Upload a car image (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear, front-facing or side-facing photo for best results.",
)

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img    = Image.fromarray(img_rgb)

    col_img, col_info = st.columns([1.2, 1])

    with col_img:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with col_info:
        st.markdown("**Image Details**")
        h, w = img_bgr.shape[:2]
        size_kb = len(file_bytes) / 1024

        st.markdown(f"""
<div class="metric-card">
  <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px">Resolution</div>
  <div style="font-size:1.4rem;font-family:'Syne',sans-serif;font-weight:700">{w} × {h}</div>
</div>
<div class="metric-card">
  <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px">File Size</div>
  <div style="font-size:1.4rem;font-family:'Syne',sans-serif;font-weight:700">{size_kb:.1f} KB</div>
</div>
<div class="metric-card">
  <div style="font-size:0.75rem;opacity:0.6;text-transform:uppercase;letter-spacing:1px">Format</div>
  <div style="font-size:1.4rem;font-family:'Syne',sans-serif;font-weight:700">{uploaded.type.split("/")[-1].upper()}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    if model is None:
        st.info("Train and add the model to enable live predictions.")
    else:
        if st.button("🔍  Analyse Image"):
            with st.spinner("Running CNN inference…"):
                time.sleep(0.4)   # slight delay for UX feel
                label, probs = predict(model, img_bgr)

            damaged_conf   = float(probs[0]) * 100
            undamaged_conf = float(probs[1]) * 100

            if label == 0:
                st.markdown(f"""
<div class="result-damaged">
  <div class="result-title" style="color:#e85d04">⚠️ DAMAGE DETECTED</div>
  <p class="result-subtitle">The model identified visible damage on this vehicle.</p>
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill-dmg" style="width:{damaged_conf:.1f}%"></div>
  </div>
  <div style="font-size:2rem;font-family:'Syne',sans-serif;font-weight:800;color:#f48c06">{damaged_conf:.1f}%</div>
  <div style="opacity:0.6;font-size:0.85rem">Confidence</div>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-clean">
  <div class="result-title" style="color:#2dc653">✅ NO DAMAGE FOUND</div>
  <p class="result-subtitle">The vehicle appears to be in good condition.</p>
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill-ok" style="width:{undamaged_conf:.1f}%"></div>
  </div>
  <div style="font-size:2rem;font-family:'Syne',sans-serif;font-weight:800;color:#80ed99">{undamaged_conf:.1f}%</div>
  <div style="opacity:0.6;font-size:0.85rem">Confidence</div>
</div>
""", unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("#### Probability Breakdown")
            c1, c2 = st.columns(2)
            c1.metric("🔴 Damaged",    f"{damaged_conf:.2f}%")
            c2.metric("🟢 Undamaged",  f"{undamaged_conf:.2f}%")

            st.markdown("""
<div class='info-box'>
Confidence below 70% may indicate an ambiguous image.
Try a clearer or differently angled photo.
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div style="text-align:center;padding:60px 20px;opacity:0.4;">
  <div style="font-size:4rem">📷</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.1rem;margin-top:12px">
    Upload a car image to get started
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;opacity:0.4;font-size:0.8rem'>"
    "Built with TensorFlow · OpenCV · Streamlit &nbsp;|&nbsp; "
    "Dataset: <a href='https://www.kaggle.com/datasets/anujms/car-damage-detection' "
    "style='color:inherit'>Kaggle — Car Damage Detection</a>"
    "</div>",
    unsafe_allow_html=True,
)
