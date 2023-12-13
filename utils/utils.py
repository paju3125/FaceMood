import os
import tempfile

import cv2 as cv
import numpy as np
import mediapipe as mp
import streamlit as st
import tensorflow as tf

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from .modelutils import my_new_model

from tensorflow.keras.utils import get_file


# emotions dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load the cascade file
face_cascade = cv.CascadeClassifier('./haarcascade_frontalface_alt.xml')

model_url = 'https://fer-model.s3.ap-south-1.amazonaws.com/base_1_overfit.h5'


# load mediapipe model and drawing utils
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
model_type = None


# get model on the choice of the user
def get_models():
    global model_type
    model_type = st.sidebar.selectbox('Model', ('Model with VGG', 'Model with VGG 3', 'Model 3 LSTM'))

    # load emotion model
    if model_type == 'Model with VGG':
        model_path = get_file('base_1_overfit.h5', model_url, cache_subdir='models')
        model = tf.keras.models.load_model(model_path)
        print(model.summary())
        return model
    elif model_type == 'Model with VGG 3':
        return tf.keras.models.load_model('./models/emotion_recognition_model.h5')
    else:
        return tf.keras.models.load_model('./models/model_1.h5')


def rescale():
    return model_type == 'Model with VGG'


# Image enhancement options
def enhance_image(image):
    # Apply sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    enhanced_image = cv.filter2D(image, -1, kernel)

    return enhanced_image

def enhance_image_with_histogram_equalization(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    enhanced_image = cv.equalizeHist(gray)
    return cv.cvtColor(enhanced_image, cv.COLOR_GRAY2BGR)

def enhance_image_with_adaptive_histogram_equalization(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray)
    return cv.cvtColor(enhanced_image, cv.COLOR_GRAY2BGR)

def enhance_image_with_unsharp_masking(image):
    blurred = cv.GaussianBlur(image, (0, 0), 10)
    enhanced_image = cv.addWeighted(image, 1.5, blurred, -0.5, 0)
    return enhanced_image

def enhance_image_with_bilateral_filter(image):
    enhanced_image = cv.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return enhanced_image

def get_image_file():
    img_file_buffer = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    st.divider()

    # Check if image is uploaded or not
    if img_file_buffer is not None:
        # read uploaded image
        image_file = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        return cv.imdecode(image_file, 1)
    else:
        # read demo image
        demo_image = "./assets/single face 2.jpg"
        return cv.imread(demo_image)


def get_video_file(use_webcam):
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcam:
            video = cv.VideoCapture(0)
        else:
            video = None
    else:
        temp_file.write(video_file_buffer.read())
        video = cv.VideoCapture(temp_file.name)

    return video


def mediapipe_detection(detection_confidence, image, model, mode):
    with mp_face_detection.FaceDetection(model_selection=1,
                                         min_detection_confidence=detection_confidence) as face_detection:
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    image_rows, image_cols, _ = image.shape
    out_img = image.copy()

    if results.detections:
        for detection in results.detections:
            try:
                # Draw face detection box
                mp_drawing.draw_detection(out_img, detection)

                box = detection.location_data.relative_bounding_box
                x = _normalized_to_pixel_coordinates(box.xmin, box.ymin, image_cols, image_rows)
                y = _normalized_to_pixel_coordinates(box.xmin + box.width, box.ymin + box.height, image_cols,
                                                     image_rows)

                # Crop image to face
                cimg = image[x[1] - 20:y[1] + 20, x[0] - 20:y[0] + 20]
                if rescale():
                    cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
                else:
                    cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0) / 255.

                # get model prediction
                pred = model.predict(cropped_img)
                idx = int(np.argmax(pred))

                cv.putText(out_img, emotion_dict[idx], (x[0], x[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                           cv.LINE_AA)

                if mode == 'With cropped image':
                    st.write('Emotion: ', emotion_dict[idx])
                    st.image(cv.resize(cimg, (300, 300)), channels="BGR", caption='Cropped Image')

            except Exception:
                pass

        if mode == 'With full image':
            st.image(out_img, channels="BGR", use_column_width=True)


def opencv_detection(image, model, mode):
    out_img = image.copy()

    # Detect the faces
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(out_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop image to face
        cimg = image[y:y + h, x:x + w]
        if rescale():
            cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0)
        else:
            cropped_img = np.expand_dims(cv.resize(cimg, (48, 48)), 0) / 255.

        # get model prediction
        pred = model.predict(cropped_img)
        idx = int(np.argmax(pred))

        cv.putText(out_img, emotion_dict[idx], (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)

        if mode == 'With cropped image':
            st.write('Emotion: ', emotion_dict[idx])
            st.image(cv.resize(cimg, (300, 300)), channels="BGR", caption='Cropped Image')

    if mode == 'With full image':
        st.image(out_img, channels="BGR", use_column_width=True)


