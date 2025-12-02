import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # hide TF warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# --- Load the trained model ---
best_model_path = "C:/Users/Malathi M/OneDrive/Documents/MDTE25/capstone project/Project 5/best_fish_model_MobileNetV2.h5"
model = load_model(best_model_path, compile=False)   # avoid compile warnings

# --- Load metadata ---
with open("C:/Users/Malathi M/OneDrive/Documents/MDTE25/capstone project/Project 5/best_model_info.pkl", "rb") as f:
    metadata = pickle.load(f)

# --- Class Names ---
class_names = [
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

st.title("🐟 Fish Image Classification")
st.write("Upload a fish image and get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Display image (NEW STREAMLIT 2025 FORMAT)
    st.image(img, caption="Uploaded Image", width="stretch")

    # Make prediction
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    st.subheader("🎯 Prediction Result")
    st.write(f"**Predicted Class:** {class_names[class_idx]}")
    st.write(f"**Confidence:** `{confidence * 100:.2f}%`")

    st.subheader("📊 All Class Probabilities")
    for class_name, prob in zip(class_names, predictions[0]):
        st.write(f"**{class_name}:** {prob * 100:.2f}%")



