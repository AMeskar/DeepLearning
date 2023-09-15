import timeit
import zipfile
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.preprocessing import normalize
import numpy as np
import os
from PIL import Image


def load_npz(dir_path):
    c = 0
    images = []

    list_of_files = os.listdir(dir_path)
    for filename in list_of_files:
        if c <= 100000:
            file_path = os.path.join(dir_path, filename)
            data = np.load(file_path)
            matrix_keys = list(data.files)

            for key in matrix_keys:
                matrix = data[key]
                num_rows = matrix.shape[0]
                max_vals = np.max(matrix, axis=0)

                if np.any(max_vals == 0):
                    normalized_matrix = np.where(max_vals == 0, matrix, matrix / max_vals)
                else:
                    normalized_matrix = matrix / max_vals

                if 1 <= num_rows <= 10000:
                    pad_width = ((0, 10000 - num_rows), (0, 0))
                    padded_matrix = np.pad(normalized_matrix, pad_width, mode='constant')

                    # Creating image from the normalized matrix
                    image = Image.fromarray((padded_matrix * 255).astype(np.uint8))
                    images.append(image)
                    c += 1

            data.close()

    if len(images) == 0:
        raise ValueError("No matrices with 50 to 100 rows found.")

    # Reshape the images and add the channel dimension
    reshaped_images = np.array([np.array(image) for image in images])
    reshaped_images = np.expand_dims(reshaped_images, axis=-1)

    return reshaped_images


dir1 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_CC'
dir2 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_NC'
dir3 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_CC'
dir4 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_NC'
dir5 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_CC'
dir6 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_NC'
dir7 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_CC'
dir8 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_NC'



images1 = load_npz(dir1)
images2 = load_npz(dir2)
images3 = load_npz(dir3)
images4 = load_npz(dir4)
images5 = load_npz(dir5)
images6 = load_npz(dir6)
images7 = load_npz(dir7)
images8 = load_npz(dir8)

# Display the shape of the loaded images
print("Images 1 shape:", images1.shape)
print("Images 2 shape:", images2.shape)
print("Images 3 shape:", images3.shape)
print("Images 4 shape:", images4.shape)
print("Images 5 shape:", images5.shape)
print("Images 6 shape:", images6.shape)
print("Images 7 shape:", images7.shape)
print("Images 8 shape:", images8.shape)

npy_array = np.concatenate((images1, images2, images3, images4,
                           images5, images6, images7, images8), axis=0)

print(np.shape(npy_array))

label1 = np.zeros(len(images1))
label2 = np.ones(len(images2))
label3 = np.full(len(images3), 2)
label4 = np.full(len(images4), 3)
label5 = np.full(len(images5), 4)
label6 = np.full(len(images6), 5)
label7 = np.full(len(images7), 6)
label8 = np.full(len(images8), 7)

labels = np.concatenate([label1, label2, label3, label4, label5, label6, label7, label8], axis=0)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=8)
print(np.shape(labels))



X_train, X_test, y_train, y_test = train_test_split(
    npy_array, labels, test_size=0.3, random_state=42)

#X_train = X_train.reshape(X_train.shape[0], -1)
#X_test = X_test.reshape(X_test.shape[0], -1)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_test))
print(np.shape(y_train))
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(100, 7, 1), padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))  # Add a dropout layer with a dropout rate of 0.2
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss = "categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=5, epochs=10,
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print('Training loss:', loss)
print('Training accuracy:', accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('annaccuracy.png')

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('annloss.png')
