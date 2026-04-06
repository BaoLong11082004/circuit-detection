import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Circuit Detection", layout="centered")

st.title("🔌 Circuit Component Detection")

# Load model (cache cho nhanh)
@st.cache_resource
def load_model():
    return YOLO("curuit_board.pt")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📷 Input", use_column_width=True)

    # Predict
    results = model.predict(img_array, conf=0.25, iou=0.7)

    result_img = results[0].plot()

    with col2:
        st.image(result_img, caption="🎯 Detected", use_column_width=True)

    boxes = results[0].boxes
    total = len(boxs) if boxes is not None else 0

    st.markdown(f"## 🔢 Total Components: `{total}`")

    names = model.names
    class_counts = {}

    if boxes is not None:
        for cls in boxes.cls:
            name = names[int(cls)]
            class_counts[name] = class_counts.get(name, 0) + 1

    st.markdown("## 📊 Component Details")
    st.json(class_counts)
