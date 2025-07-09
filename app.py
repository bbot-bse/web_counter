import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io

st.set_page_config(page_title="Roboflow Detector", layout="centered")
st.title("📸 Roboflow Object Detection")
st.write("Upload an image to detect objects using your Roboflow model.")

# Load Roboflow model
rf = Roboflow(api_key=st.secrets["ROBOFLOW_API_KEY"])
project = rf.workspace().project("your-project-name")  # Replace with your project name
model = project.version(1).model                       # Replace with your version if not v1

# Upload file
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_bytes = uploaded_file.read()
    
    with st.spinner("Running detection..."):
        try:
            prediction = model.predict(image_bytes).json()
            st.success("Detection complete!")
            st.json(prediction)
        except Exception as e:
            st.error(f"Error: {str(e)}")
