import streamlit as st
from PIL import Image
import numpy as np 
import tensorflow as tf 

def detect_age_gender(image):
    Model = tf.keras.models.load_model('/home/user/Documents/ML_DL_PROJECTS/GenderAgeDetectorCVProjectByNullClass/Age_N_Gender_Detector.h5')
    image = image.resize((48, 48))
    # image = np.expand_dims(image, axis=0)
    image = np.array(image)
    # image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    image = np.array([image]) / 255 
    predicted = Model.predict(image)
    age = int(np.round(predicted[1][0]))
    gender = int(np.round(predicted[0][0]))
    return age, gender

st.set_page_config(page_title='Age & Gender Detection Project CV Project', layout='centered')
st.title('Age & Gender Detection')
st.header('Trained & Developed by Sai Kiran Patnana')


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',use_column_width=True)
    age, gender = detect_age_gender(image)
    
    st.markdown(f"<h3 style='text-align: center;'>Predicted Age: {age}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>Predicted Gender: {'Male' if gender == 0 else 'Female'}</h3>", unsafe_allow_html=True)
  

    

