import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# -------------------------------
# Load trained CNN model
# -------------------------------
model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    layout="centered"
)

st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload an image and the CNN model will automatically predict its class.")

st.sidebar.title("About")
st.sidebar.info(
    "This app uses a CNN trained on the CIFAR-10 dataset.\n\n"
    "⚠️ CIFAR-10 images are 32×32 pixels, so real-world images may have lower accuracy."
)

# -------------------------------
# Image uploader
# -------------------------------
uploaded_image = st.file_uploader(
    "Upload an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction pipeline
# -------------------------------
if uploaded_image is not None:
    # Load and display image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to NumPy array
    img = np.array(image)

    # Handle RGBA images
    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize to CIFAR-10 size
    img = cv2.resize(img, (32, 32))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    # -------------------------------
    # Show results
    # -------------------------------
    st.subheader("🔮 Prediction Result")
    st.success(f"**Predicted Class:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Probability chart
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": prediction[0]
    })

    st.subheader("📊 Class Probabilities")
    st.bar_chart(prob_df.set_index("Class"))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption(
    "Built using TensorFlow, CNN, and Streamlit | Microsoft Edunet Project"
)
