import streamlit as st
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="Roboflow Detector", layout="centered")
st.title("📸 Roboflow Object Detection")
st.write("Upload an image and adjust the confidence threshold below:")

# Slider for confidence threshold
confidence_threshold = st.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05
)

# Initialize Roboflow model
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project = rf.workspace().project("flower-counter")     # ← Replace
model = project.version(11).model                          # ← Replace if needed

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded Image", use_container_width=True)

    # Convert to byte stream
    img_bytes = BytesIO()
    image_pil.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with st.spinner("Running detection..."):
        try:
            result = model.predict(img_bytes).json()
            predictions = result["predictions"]

            # Filter by threshold
            filtered = [p for p in predictions if p["confidence"] >= confidence_threshold]

            # Draw boxes
            annotated = image_pil.copy()
            draw = ImageDraw.Draw(annotated)
            font = ImageFont.load_default()
            label_counts = {}

            for pred in filtered:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                cls = pred["class"]
                conf = pred["confidence"]
                label = f"{cls} ({conf:.2f})"

                # Calculate box corners
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                draw.text((x1, y1 - 10), label, fill="lime", font=font)

                label_counts[cls] = label_counts.get(cls, 0) + 1

            # Show image
            st.image(annotated, caption="Detections", use_container_width=True)

            # Show detection summary
            if label_counts:
                st.subheader("📊 Detection Summary")
                df_summary = pd.DataFrame(list(label_counts.items()), columns=["Class", "Count"])
                st.table(df_summary)

            # Download button
            out_buffer = BytesIO()
            annotated.save(out_buffer, format="PNG")
            st.download_button(
                label="📥 Download Annotated Image",
                data=out_buffer.getvalue(),
                file_name="detections.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Detection failed: {e}")
