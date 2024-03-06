import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

all_categories = [
    'faceshield',
    'facemask',
    'glasses',
    'gloves',
    'hairnet',
    'hospital_bed',
    'medical_instrument',
    'monitor',
    'operatinglights',
    'scrubs'
]

class_mapping = {category: index for index, category in enumerate(sorted(all_categories))}
numeric_labels = list(range(len(all_categories)))

from tensorflow.keras.models import load_model
model = load_model('C:/Users/johnm/images_project/inception_final.h5')

inverse_class_mapping = {v: k for k, v in class_mapping.items()}

def make_img_pred(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_fix = img_array.reshape(1, 224, 224, 3)
    img_fixed = img_fix.astype('float32')
    result = model.predict(img_fixed)
    predicted_class_index = np.argmax(result)
    predicted_class_name = inverse_class_mapping[predicted_class_index]
    st.image(img, caption=f"Uploaded Image - Predicted Class: {predicted_class_name}", use_column_width=True)
    st.write(f"The model predicted class: {predicted_class_name}")

# Streamlit UI
st.balloons()
st.info("This is a medical classification prediction app. It will predict medical object classes. Please upload a photo you want the classification model to predict on")
st.sidebar.title("John Munoz\n\nDeep Learning Class - 3/05/2024")
st.sidebar.info("In this Streamlit app, you can predict medical object classes using a pre-trained model.")
st.sidebar.write("These are the ten medical objects the Model Can Predict on")
st.sidebar.write(all_categories)

st.markdown("<h1 class='stTitle'> Medical Object Classification </h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="image_upload")

if uploaded_file is not None:
    make_img_pred(uploaded_file)
