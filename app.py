import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model_path = "C:/Users/BHARGAV/Downloads/road-dataset"  # Update this with your actual model path
model = tf.keras.models.load_model(model_path)

# Define IOU metric function
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

# Function to preprocess and predict on user-uploaded image
def predict_image(image):
    image = np.array(image.resize((256, 256))) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction[0, ..., 0]

# Streamlit app code
st.title("Road Extraction Model")
st.write("Upload an image to predict road extraction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict_image(image)
        st.image(prediction, caption="Predicted Mask", use_column_width=True)
