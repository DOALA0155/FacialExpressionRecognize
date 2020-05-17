import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_image(pixels):
    plt.imshow(pixels)
    plt.show()

with open("./Data/image_data.pkl", "rb") as f:
    image_data = pickle.load(f)

image_data = np.zeros((48, 48, 3))
image = image_data[0]
for i, row in enumerate(image):
    for j, column in enumerate(row):

plot_image(image_data)
