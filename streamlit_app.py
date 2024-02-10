import streamlit as st
from PIL import Image
import numpy as np 
import tensorflow as tf 

def detect_age_gender(image):
    Model = tf.keras.models.load_model('/home/user/Documents/ML_DL_PROJECTS/GenderAgeDetectorCVProjectByNullClass/Age_N_Gender_Detector.h5')
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    print(image.shape)
    image = np.array([image]) / 255 
    predicted = Model.predict(image)
    age = int(np.round(predicted[1][0]))
    gender = int(np.round(predicted[0][0]))
    print('Predicted Age: ', str(age))
    print('Predicted Gender: ', str(gender))
    return age, gender

st.title('Age & Gender Detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=400)
    age, gender = detect_age_gender(image)
    st.write(f"Predicted Age: {age}")
    st.write(f"Predicted Gender: {'Male' if gender == 0 else 'Female'}")
