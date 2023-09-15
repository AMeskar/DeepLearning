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


def load_npz(dir_path):
    c =0
    concatenated_matrix = None
    
    list_of_files = os.listdir(dir_path)
    for filename in list_of_files:
       if c <= 10000:
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

            if 50 <= num_rows <= 100:
                pad_width = ((0, 100 - num_rows), (0, 0))
                padded_matrix = np.pad(normalized_matrix, pad_width, mode='constant')
                expanded_matrix = np.expand_dims(padded_matrix, axis=0)

                if concatenated_matrix is None:
                    concatenated_matrix = expanded_matrix
                else:
                    concatenated_matrix = np.concatenate((concatenated_matrix, expanded_matrix), axis=0)
                c += 1
        data.close()

    if concatenated_matrix is None:
        raise ValueError("No matrices with 50 to 100 rows found.")

    return concatenated_matrix
    
dir1 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_CC'
dir2 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_NC'
dir3 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_CC'
dir4 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_NC'
dir5 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_CC'
dir6 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_NC'
dir7 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_CC'
dir8 = '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_NC'


matrix1 = load_npz(dir1)
matrix2 = load_npz(dir2)
matrix3 = load_npz(dir3)
matrix4 = load_npz(dir4)
matrix5 = load_npz(dir5)
matrix6 = load_npz(dir6)
matrix7 = load_npz(dir7)
matrix8 = load_npz(dir8)



print(np.shape(matrix1))
print(np.shape(matrix2))
print(np.shape(matrix3))
print(np.shape(matrix4))
print(np.shape(matrix5))
print(np.shape(matrix6))
print(np.shape(matrix7))
print(np.shape(matrix8))


npy_array = np.concatenate((matrix1, matrix2, matrix3, matrix4,
                           matrix5, matrix6, matrix7, matrix8), axis=0)

print(np.shape(npy_array))

label1= np.zeros(len(matrix1))
label2= np.ones(len(matrix2))
label3= np.full(len(matrix3), 2)
label4= np.full(len(matrix4), 3)
label5= np.full(len(matrix5), 4)
label6= np.full(len(matrix6), 5)
label7= np.full(len(matrix7), 6)
label8= np.full(len(matrix8), 7)

labels= np.concatenate([label1, label2, label3, label4, label5, label6, label7, label8], axis= 0)

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=8)
print(np.shape(labels))


X_train, X_test, y_train, y_test = train_test_split(
    npy_array, labels, test_size=0.3, random_state=42)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print(np.shape(X_train))
print(np.shape(X_test))
print(np.shape(y_test))
print(np.shape(y_train))

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=X_train[0].shape), 
    layers.BatchNormalization(), # Input layer
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
   # layers.Dense(128, activation='relu'), 
    #layers.BatchNormalization(), 
    layers.Dropout(0.4), # Hidden layer
    layers.Dense(8, activation='softmax')])  # Output layer

model.compile(optimizer='adam',
              loss = "categorical_crossentropy",
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=2, epochs=10,
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
