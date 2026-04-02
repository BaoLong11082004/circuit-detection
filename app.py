import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json

# Load model
model = YOLO("best.pt")

st.title("🔌 Circuit Component Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image")

    results = model.predict(image, conf=0.25, iou=0.7)

    result_img = results[0].plot()
    st.image(result_img, caption="Detected")

    boxes = results[0].boxes
    total = len(boxes)

    st.write(f"### 🔢 Total: {total}")

    names = model.names
    class_counts = {}

    for cls in boxes.cls:
        name = names[int(cls)]
        class_counts[name] = class_counts.get(name, 0) + 1

    st.write("### 📊 Details")
    st.json(class_counts)
