import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from plot_history import plot

with open("./Data/image_data.pkl", "rb") as f:
    image_data = pickle.load(f)

with open("./Data/emotion_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

labels = label_data.unique()
print(labels)

label_data = to_categorical(label_data)
x_train, x_test, y_train, y_test = train_test_split(image_data, label_data, shuffle=True)

model = Sequential()
model.add(Conv2D(120, kernel_size=(2, 2), activation="relu", input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, kernel_size=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation="softmax"))

print(model.summary())

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
plot(history)

model.save("./Models/FER_CNN.h5")
