import cv2
import time
import tempfile
import argparse
import streamlit as st
import onnxruntime as ort
from PIL import Image
from helpers import predict_image

# Set Argument Parse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--device",
    type=str,
    default="CPU",
    help="Device to use for inference (CPU or CUDA)",
)
parser.add_argument(
    "-",
    "--model",
    type=str,
    default="./model_binary.onnx",
    help="Path to model",
)
value_parser = parser.parse_args()

# Define constant variable
DEVICE_INFERENCE = value_parser.device
MODEL_PATH = value_parser.model

# Set page config
st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🕵️",
)

# Load model
@st.cache(allow_output_mutation=True)
def load_model(model_path, device_inference="cpu"):
    if device_inference.lower() == "cpu":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    elif device_inference.lower() == "cuda":
        ort_session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
        )
    else:
        st.error("Please select between CPU or CUDA!")
        st.stop()

    return ort_session


# Run load model
model = load_model(MODEL_PATH, DEVICE_INFERENCE)

# Main page
st.title("Steel Defect Detection")
st.write(
    """
        Usually, there is some trouble in production machines and steel defects because of heating and rolling, drying and cutting, several machines touch flat steel. Because of that, steel has been defected from production. The purpose of this project is to give an image of steel, after that we need to detect segmentation defects in steel.
"""
)
st.markdown("  ")

format_file = st.selectbox("Select format file to predict", ["Image", "Video"])
if format_file.lower() == "image":
    uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file = Image.open(uploaded_file)
        st.markdown("  ")
        st.write("Source Image")
        st.image(uploaded_file)

        predict_button = st.button("Detect steel defect")
        st.markdown("  ")

        if predict_button:
            with st.spinner("Wait for it..."):
                start_time = time.time()
                mask_image, segmentation_image = predict_image(uploaded_file, model)
                st.write("Mask Image")
                st.image(mask_image)
                st.write("Segmentation Image")
                st.image(segmentation_image)
                st.write(f"Inference time: {(time.time() - start_time):.3f} seconds")

elif format_file.lower() == "video":
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        st.markdown("  ")
        st.write("Source Video")
        st.video(uploaded_file)
        predict_button = st.button("Detect forest fire")
        st.markdown("  ")
        if predict_button:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                cap = cv2.VideoCapture(tmp.name)
                st.write("Mask Image")
                mask_image_empty = st.empty()
                st.write("Segmentation Image")
                segmentation_image_empty = st.empty()
                while cap.isOpened():
                    _, frame = cap.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    mask_image, segmentation_image = predict_image(frame, model)
                    mask_image_empty.image(
                        mask_image,
                    )
                    segmentation_image_empty.image(
                        segmentation_image,
                    )
                cap.release()
