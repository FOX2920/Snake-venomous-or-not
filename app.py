import streamlit as st
from PIL import Image
import numpy as np
import urllib.request
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
def predict(img_array):
    result = model.predict(img_array)
    return result

# Streamlit app
def main():
    st.title("Nhận Diện rắn độc Hay không độc ")

    uploaded_file = st.file_uploader("Chọn ảnh rắn hoặc nhập URL...", type=["jpg", "jpeg", "png"])
    image_path = None

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_column_width=True)
        image_path = './uploaded_image.jpg'
        image.save(image_path)

    url = st.text_input('Hoặc nhập URL ảnh rắn:', 'https://th.bing.com/th/id/OIP.KUwhBZXNmh5-4lfoQY-LSAHaEl?rs=1&pid=ImgDetMain')
    
    if url:
        urllib.request.urlretrieve(url, './url_image.jpg')
        st.image(url, caption="Ảnh từ URL", use_column_width=True)
        image_path = './url_image.jpg'

    if image_path is not None:
        # Make predictions
        img_array = preprocess_image(image_path)
        prediction = predict(img_array)

        # Display the result
        if prediction[0][0] > 0.5:
            st.subheader("Dự đoán: Rắn có độc")
        else:
            st.subheader("Dự đoán: Rắn không độc")

if __name__ == "__main__":
    main()
