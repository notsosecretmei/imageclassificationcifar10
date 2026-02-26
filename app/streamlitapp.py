import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Cache the model to avoid reloading on every run
@st.cache_resource
def load_cifar10_model(model_path):
    return load_model(model_path)

# Load the model
model_path = os.path.join("model", "cifar10model.h5")
model = load_cifar10_model(model_path)

# CIFAR-10 classes
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# App title and description
st.title("CIFAR-10 Image Classifier")
st.markdown("**Purpose:** Deploy a simple image classification model interactively using Streamlit.")
st.write("Upload an image, and the model will predict its class among 10 CIFAR-10 categories.")

# Warning about limitations
st.warning(
    "⚠️ Note: This model performs best on simple images similar to the CIFAR-10 dataset "
    "(single objects like cats, dogs, airplanes, etc.). Complex backgrounds may be misclassified."
)

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    # Open and convert image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resize to 32x32
    image = image.resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prepare image for model
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    prediction = model.predict(image_array, verbose=0)
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    
    # Confidence check
    confidence_threshold = 0.7
    if confidence < confidence_threshold:
        st.error(
            "❌ Image not recognized confidently. "
            "Please upload a simple CIFAR-10 object (like a cat, dog, airplane, etc.)."
        )
    else:
        st.success(f"Prediction: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence:.2f}")