import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("cifar10_resnet50_model.keras")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.set_page_config(
    page_title="CIFAR-10 Object Recognition",
    page_icon="🖼️"
)

st.title("🖼️ CIFAR-10 Object Recognition using ResNet50")

st.write(
    "Upload an image and the model will classify it into one of the CIFAR-10 object categories."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess image
    image_resized = image.resize((32, 32))
    image_array = np.array(image_resized)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0])

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class_name}")
    st.write(f"Confidence Score: {confidence:.2f}")

    # Show all class probabilities
    st.subheader("Class Probabilities")

    probabilities = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }

    st.bar_chart(probabilities)