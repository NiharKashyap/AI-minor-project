import cv2
import streamlit as st
from PIL import Image

import numpy as np
import os


from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
# Load the model

model = tf.keras.models.load_model('keras_model.h5')


def saveImg(uploaded_file):
    if uploaded_file is not None:
     with open(os.path.join("tempDir",uploaded_file.name),"wb") as f: 
         f.write(uploaded_file.getbuffer())
    
    return os.path.join("tempDir",uploaded_file.name)
		


def detect(path):
    classes = ['Corn(maize) Common rust','Corn(maize) healthy','Corn(maize) Northern Leaf Blight','Potato Early blight','Potato healthy', 'Potato Late blight']
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(path)
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    label = np.argmax(prediction)
    print(prediction)
    print(classes[label])
    return classes[label]


uploaded_file = st.file_uploader("Choose a Image file")



if uploaded_file is not None:
    frame = saveImg(uploaded_file)
    pred = detect(frame)
    imgS = Image.open(frame)
    st.image(imgS)
    st.header('Predicted disease: ' + pred)