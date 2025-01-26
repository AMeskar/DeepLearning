#################################################################################
################# LIBRARIES #####################################################
#################################################################################

import timeit
import zipfile
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_npz(dir_path, max_images=10000):
    c = 0
    concatenated_matrix = None

    for filename in sorted(os.listdir(dir_path)):
        if c >= max_images:
            break

        file_path = os.path.join(dir_path, filename)
        data = np.load(file_path)
        matrix_keys = list(data.files)

        for key in matrix_keys:
            if c >= max_images:
                break

            matrix = data[key]
            num_rows = matrix.shape[0]
            max_vals = np.max(matrix, axis=0)

            # Normalize, avoiding division by zero
            normalized_matrix = np.where(max_vals == 0, matrix, matrix / max_vals)

            # Keep only matrices with 50 <= rows <= 100
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
        raise ValueError(f"No valid matrices found in {dir_path}")

    return concatenated_matrix

#################################################################################
################# DIRECTORIES ###################################################
#################################################################################

dir_paths = [
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_CC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anue_b_NC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_CC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/anumu_b_NC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_CC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/nue_b_NC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_CC',
    '/sps/atlas/h/hatmani/ANTARES/Inputs/Matrix/numu_b_NC'
]

#################################################################################
################# LOAD IMAGES ###################################################
#################################################################################

matrix_list = [load_npz(path, max_images=10000) for path in dir_paths]

for i, mat in enumerate(matrix_list, start=1):
    print(f"Matrix {i} shape: {mat.shape}")

all_data = np.concatenate(matrix_list, axis=0)
print("All data shape:", all_data.shape)

all_labels = []
for class_idx, mat in enumerate(matrix_list):
    all_labels.append(np.full(len(mat), class_idx))
all_labels = np.concatenate(all_labels)
all_labels = to_categorical(all_labels, num_classes=len(dir_paths))
print("All labels shape:", all_labels.shape)

#################################################################################
################# TRAIN/TEST SPLIT ##############################################
#################################################################################

X_train, X_test, y_train, y_test = train_test_split(
    all_data, all_labels, test_size=0.3, random_state=42
)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


#################################################################################
################# MODEL #########################################################
#################################################################################

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#################################################################################
################# EVALUATION ####################################################
#################################################################################

history = model.fit(
    X_train, y_train,
    batch_size=2,
    epochs=10,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#################################################################################
############# PLOTTING ##########################################################
#################################################################################

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('ann_accuracy.png')

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ann_loss.png')
plt.show()
