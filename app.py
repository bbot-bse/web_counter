import streamlit as st
from roboflow import Roboflow
import numpy as np
import cv2
from PIL import Image

# SAHI imports
from sahi.model import RemoteInferenceModel
from sahi.predict import get_sliced_prediction
from sahi.utils.visualization import visualize_image_prediction

st.set_page_config(page_title="Roboflow + SAHI Detector", layout="centered")
st.title("📸 Roboflow Object Detection with SAHI Slicing")
st.write("Upload an image; it will be sliced into 640×640 patches for inference.")

#
# 1) Initialize Roboflow model (for metadata) and get its endpoint URL
#
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project = rf.workspace().project("flower-counter")   # ← replace
model_version = project.version(11)                      # ← replace if not v1
rf_endpoint = model_version.model.url                  # this is the full HTTP endpoint

#
# 2) Wrap it in a SAHI remote model with 640×640 slicing
#
sahi_model = RemoteInferenceModel(
    model_type="yolov8",                # Roboflow exports YOLOv8 under the hood
    endpoint_url=rf_endpoint,
    confidence_threshold=0.3,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,           # adjust overlap if you like
    overlap_width_ratio=0.2,
    device="cpu"                        # or "cuda:0" if GPU available
)

#
# 3) Uploader + inference + visualization
#
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    # display original
    st.image(uploaded, caption="Original Image", use_container_width=True)

    # convert to OpenCV BGR numpy
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with st.spinner("Running sliced inference..."):
        try:
            # get SAHI prediction
            prediction = get_sliced_prediction(
                img_bgr,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )

            # visualize merged prediction on the full image
            vis_bgr = visualize_image_prediction(
                img_bgr,
                prediction,
                show_bbox=True,
                show_label=True,
                show_confidence=True,
                line_thickness=2,
            )

            # convert back to RGB for Streamlit
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption="Detections (SAHI)", use_container_width=True)

        except Exception as e:
            st.error(f"Inference error: {e}")
