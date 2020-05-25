import pandas as pd
import pickle
import numpy as np
import pyprind
import cv2
import face_recognition
import matplotlib.pyplot as plt
import time

def preprocessing_images():
    data_dir = "/Users/katsuyamashouki/Desktop/Programming/Python/AI/Datasets/ImageData/FER2013/fer2013.csv"

    face_data = pd.read_csv(data_dir)

    emotion_labels = face_data["emotion"]
    image_data = face_data["pixels"]

    images = []
    bar = pyprind.ProgBar(len(image_data))

    for i, image in enumerate(image_data):
        image_pixels = []
        pixels = image.split(" ")
        for pixel in pixels:
            image_pixels.append(np.float32(pixel))

        image_pixels = np.array(image_pixels).reshape(48, 48, 1).astype(np.uint8)

        face_locations = face_recognition.face_locations(image_pixels)

        if len(face_locations) == 0:
            face_image = cv2.resize(image_pixels, (350, 350))
        else:
            face_location = face_locations[0]
            top, right, bottom, left = face_location
            face_image = image_pixels[top: bottom, left: right]
            face_image = cv2.resize(face_image, (350, 350))

        color_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB).astype(np.float16)
        color_image /= 255.
        color_image = color_image.astype(np.float16)
        images.append(color_image)
        bar.update()

    images = np.array(images)

    with open("./Data/emotion_labels.pkl", "wb") as f:
        pickle.dump(emotion_labels, f)

    with open("./Data/image_data.pkl", "wb") as f:
        pickle.dump(images, f)

def crop_face(image):
    face_locations = face_recognition.face_locations(image)
    top, bottom, left, right = face_locations
    croped_image = image[top: bottom, left: right]
    return croped_image

preprocessing_images()
