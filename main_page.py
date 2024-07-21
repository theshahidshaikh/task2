import streamlit as st
import tensorflow as tf
import numpy as np


# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("Train_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "How It Works", "About"])

# Home Page
if page == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "E:\Bharti Vidhypith\INTERNSHIP\Deep Learning projects\\bg.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
     Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    """)
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image:
        st.write("Processing your image, please wait...")
        with st.spinner('Analyzing the image...'):
            result_index = model_prediction(test_image)
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Potato___Late_blight', 'Potato___healthy',
                          'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___healthy', 'Potato___Early_Blight']

            st.success("Model is Predicting it's a {}".format(class_name[result_index]))
            st.image(test_image, use_column_width=True)

# How It Works Page
elif page == "How It Works":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "E:\Bharti Vidhypith\INTERNSHIP\Deep Learning projects\\bg.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
         Welcome to the Plant Disease Recognition System! 🌿🔍

            Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        """)
    st.header("How It Works")
    st.markdown("""
        ### How It Works
        1. **Upload Image:** Upload an image of a plant with suspected diseases by pressing on upload button or just drag and drop.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Upload an image and experience the power of our Plant Disease Recognition System!
    """)

# About Page
elif page == "About":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "E:\Bharti Vidhypith\INTERNSHIP\Deep Learning projects\\bg.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
         Welcome to the Plant Disease Recognition System! 🌿🔍

            Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        """)
    st.header("About")
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.
        This dataset consists of about 26K rgb images of healthy and diseased crop leaves which is categorized into 11 different classes.
        A new directory containing 14 test images is created later for prediction purpose.
        #### Content
        1. train (20815 images)
        2. test (14 images)
        3. validation (5203 images)
    """)

