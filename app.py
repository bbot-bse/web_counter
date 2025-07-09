import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
import pandas as pd

# SAHI imports
from sahi.model import RemoteInferenceModel
from sahi.predict import get_sliced_prediction

st.set_page_config(page_title="Roboflow + SAHI", layout="centered")
st.title("📸 Roboflow Detection with SAHI (No OpenCV)")
st.write("Upload an image and see detections using sliced inference.")

# Load Roboflow model
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project = rf.workspace().project("flower-counter")  # ← Replace with your project
model = project.version(11).model                      # ← Replace with your version

# Wrap in SAHI RemoteInferenceModel
sahi_model = RemoteInferenceModel(
    model_type="yolov8",
    endpoint_url=model.url,
    confidence_threshold=0.3,
    device="cpu"
)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection..."):
        try:
            prediction = get_sliced_prediction(
                image_pil,
                detection_model=sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            # Draw boxes on a copy
            annotated_img = image_pil.copy()
            draw = ImageDraw.Draw(annotated_img)
            font = ImageFont.load_default()

            label_counts = {}
            for det in prediction.object_prediction_list:
                box = det.bbox.to_xyxy()  # [x1, y1, x2, y2]
                class_name = det.category.name
                score = det.score.value
                label = f"{class_name} ({score:.2f})"

                draw.rectangle(box, outline="lime", width=3)
                draw.text((box[0], box[1] - 10), label, fill="lime", font=font)

                # Tally counts
                label_counts[class_name] = label_counts.get(class_name, 0) + 1

            # Show final image
            st.image(annotated_img, caption="Detections", use_container_width=True)

            # Show detection summary
            if label_counts:
                st.subheader("📊 Detection Summary")
                df_summary = pd.DataFrame(list(label_counts.items()), columns=["Class", "Count"])
                st.table(df_summary)

            # Download button
            img_buffer = BytesIO()
            annotated_img.save(img_buffer, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=img_buffer.getvalue(),
                file_name="detections.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Detection failed: {e}")
