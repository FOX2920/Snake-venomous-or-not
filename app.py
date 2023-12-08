import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import h5py
from keras.preprocessing import image
import os

# Get the path to the pre-trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model_224.h5")

# Load the pre-trained model
with h5py.File(model_path, 'r') as f:
    model = load_model(f)

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image_path):
    img_array = preprocess_image(image_path)
    result = model.predict(img_array)
    return result

# Streamlit app
def main():
    st.title("Nhận Diện rắn độc Hay không độc ")
    st.title("Nhập Ảnh")

    uploaded_file = st.file_uploader("Chọn ảnh rắn...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_column_width=True)

        # Make predictions
        prediction = predict(uploaded_file)

        # Display the result
        if prediction[0][0] > 0.5:
            st.subheader("Dự đoán: Rắn có độc")
        else:
            st.subheader("Dự đoán: Rắn không độc")

if __name__ == "__main__":
    main()