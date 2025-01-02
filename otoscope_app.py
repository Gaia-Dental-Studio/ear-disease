# app_frontend.py (Streamlit Frontend)

import streamlit as st
import requests
from PIL import Image
import io

# Flask backend URL
BACKEND_URL = "http://127.0.0.1:5000/predict"

st.title("Otoscope Image Classification")
st.write("Upload an image of the ear canal for classification (supports .jpg, .jpeg, .png, .tiff).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to bytes for sending to the backend
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes.seek(0)

    # Send the file to the Flask backend
    files = {"file": ("image.tiff", image_bytes, f"image/{image.format.lower()}")}
    response = requests.post(BACKEND_URL, files=files)

    # Display the prediction results
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Class: {result['predicted_class']}")
        st.info(f"Confidence Score: {result['confidence_score']:.2f}")

        st.write("### Class Probabilities")
        for class_name, probability in result["class_probabilities"].items():
            st.write(f"{class_name}: {probability:.2f}")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
