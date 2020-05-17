import pandas as pd
import pickle
import numpy as np
import pyprind
import cv2

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

    image_pixels = np.array(image_pixels).reshape(48, 48, 1)
    color_image = cv2.cvtColor(image_pixels, cv2.COLOR_GRAY2RGB)
    color_image /= 255.
    images.append(color_image)
    bar.update()

images = np.array(images)

with open("./Data/emotion_labels.pkl", "wb") as f:
    pickle.dump(emotion_labels, f)

with open("./Data/image_data.pkl", "wb") as f:
    pickle.dump(images, f)
