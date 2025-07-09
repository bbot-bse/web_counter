import streamlit as st
from roboflow import Roboflow
import io

# Initialize Roboflow (once)
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project = rf.workspace().project("flower-counter")
model = project.version(1).model

st.title("📸 Image Detection with Roboflow")
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Detecting..."):
        result = model.predict(img_bytes)
        st.json(result.json())
