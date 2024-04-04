import streamlit as st
import tensorflow as tf
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import IPython.display as display
import numpy as np
import glob
import random
import cv2
import shutil
from prediction_utils import pair_image_resize, get_prediction


all_disease_image_path = "alleyedisease.png"
all_disease_image = Image.open(all_disease_image_path)

if os.path.exists(os.path.join(os.getcwd(), "uploaded_img_dir")):
    shutil.rmtree(os.path.join(os.getcwd(), "uploaded_img_dir"))

os.mkdir(os.path.join(os.getcwd(), "uploaded_img_dir"))


def save_image(image_file):
    @st.cache_data
    def load_image(image_file):
        img = Image.open(image_file)
        return img

    img = load_image(image_file)

    with open(os.path.join(os.getcwd(), "uploaded_img_dir",image_file.name), "wb") as f:
        f.write(image_file.getbuffer())


add_selectbox = st.sidebar.selectbox(
    "Select ",
    ("Basic Info", "Demo")
)


# Using "with" notation
if add_selectbox == "Basic Info":
    if st.checkbox("Show/Hide: View Eye Disease Type"):
        # display the text if the checkbox returns True value
        st.image(all_disease_image)
else:
    st.header("Demo For The Client")
    input_image_type_selectbox = st.sidebar.selectbox(
        "Select input type: ",
        ("ImagePath", "Image"))
    # input_type_status = st.radio('Select input type: ',
    #                              ('ImagePath', 'Image'))

    if input_image_type_selectbox == 'ImagePath':
        st.write("Right Eye Image Path")
        right_eye = st.text_input("Enter Right Image Path Here(if present in your local)")
        st.write("Left Eye Image Path")
        left_eye = st.text_input("Enter Left Image Path Here(if present in your local)")
        submit = st.button('Get Prediction')
        if submit:
            if left_eye is not None and right_eye is not None:
                if str(os.path.basename(right_eye)).split("_")[0] == str(os.path.basename(left_eye)).split("_")[0]:
                    with st.spinner(text="This may take a moment..."):
                        st.image(Image.open(left_eye))
                        st.image(Image.open(right_eye))
                        pair_image_resize(left_eye, right_eye)
                        prediction = get_prediction()
                    st.write(prediction)
                else:
                    st.write("Enter right pair of image")
            else:
                st.write("Enter Files")
    else:
        right_eye_file = st.file_uploader("Upload Right Eye Image", type=['png', 'jpeg', 'jpg'])
        left_eye_file = st.file_uploader("Upload Left Eye Image", type=['png', 'jpeg', 'jpg'])
        if right_eye_file is not None:
            file_details = {"FileName": right_eye_file.name, "FileType": right_eye_file.type}
            st.write(file_details)
            save_image(right_eye_file)
            st.success(" Right Eye File Saved")
        if left_eye_file is not None:
            file_details = {"FileName": left_eye_file.name, "FileType": left_eye_file.type}
            st.write(file_details)
            save_image(left_eye_file)
            st.success(" left Eye File Saved")

        submit = st.button('Get Prediction')
        if submit:
            if left_eye_file is not None and right_eye_file is not None:
                if str( right_eye_file.name).split("_")[0] == str( left_eye_file.name).split("_")[0]:
                    with st.spinner(text="This may take a moment..."):
                        all_uploaded_files = os.listdir(os.path.join(os.getcwd(), "uploaded_img_dir"))
                        left_eye = os.path.join(os.getcwd(), "uploaded_img_dir", all_uploaded_files[0])
                        right_eye = os.path.join(os.getcwd(), "uploaded_img_dir", all_uploaded_files[1])
                        st.image(Image.open(left_eye))
                        st.image(Image.open(right_eye))
                        pair_image_resize(left_eye, right_eye)
                        prediction = get_prediction()
                    st.write(prediction)
                else:
                    st.write("Enter right pair of image")
            else:
                st.write("Enter Files")




