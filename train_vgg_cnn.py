import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.applications import VGG16
from sklearn.model_selection import train_test_split

with open("./Data/image_data.pkl", "rb") as f:
    image_data = pickle.load(f)

with open("./Data/emotion_labels.pkl", "rb") as f:
    label_data = pickle.load(f)

labels = label_data.unique()
print(labels)

label_data = to_categorical(label_data)
x_train, x_test, y_train, y_test = train_test_split(image_data, label_data, shuffle=True)

print(x_train.shape)
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(350, 350, 3))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(Dense(len(labels), activation="softmax"))

vgg16.trainable = False

print(model.summary())

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

filepath="./Models/model-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(x_train, y_train, epochs=20, batch_size=10, validation_split=0.2, callbacks=callbacks_list)
model.save("./Models/vgg16_cnn.h5")
