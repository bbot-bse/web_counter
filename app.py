import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import requests
import pandas as pd

st.set_page_config(page_title="Roboflow Detector", layout="centered")
st.title("📸 Roboflow Object Detection")
st.write("Upload an image and adjust the confidence threshold below:")

# Confidence slider
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

# Roboflow setup
ROBOFLOW_API_KEY = st.secrets["ROBOFLOW_API_KEY"]
MODEL_ENDPOINT = "flower-counter/12"  # ✅ Replace if needed
API_URL = f"https://detect.roboflow.com/{MODEL_ENDPOINT}?api_key={ROBOFLOW_API_KEY}"

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to JPEG bytes
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    with st.spinner("Running detection..."):
        try:
            # ❗ No headers — let requests handle Content-Type
            response = requests.post(API_URL, files={"file": buffer.getvalue()})
            result = response.json()

            if response.status_code != 200:
                st.error(f"API Error: {result.get('message', 'Unknown error')}")
            else:
                predictions = result.get("predictions", [])
                filtered = [p for p in predictions if p["confidence"] >= confidence_threshold]

                # Draw boxes
                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)
                font = ImageFont.load_default()
                label_counts = {}

                for pred in filtered:
                    x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                    cls, conf = pred["class"], pred["confidence"]
                    x1, y1 = x - w / 2, y - h / 2
                    x2, y2 = x + w / 2, y + h / 2
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                    draw.text((x1, y1 - 10), f"{cls} ({conf:.2f})", fill="lime", font=font)
                    label_counts[cls] = label_counts.get(cls, 0) + 1

                st.image(annotated, caption="Detections", use_container_width=True)

                if label_counts:
                    st.subheader("📊 Detection Summary")
                    st.table(pd.DataFrame(label_counts.items(), columns=["Class", "Count"]))

                # Download button
                output_buffer = BytesIO()
                annotated.save(output_buffer, format="PNG")
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=output_buffer.getvalue(),
                    file_name="detections.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Detection failed: {e}")
