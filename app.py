import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = os.path.join(os.path.dirname(__file__), "cifar10_cnn.keras")
model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

st.title("CIFAR-10 Image Classification (CNN)")

uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.subheader(f"Prediction: {predicted_class}")

