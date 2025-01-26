#################################################################################
################# LIBRARIES #####################################################
#################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_npz(dir_path, max_images=100000, max_rows=10000):

    images = []
    count_loaded = 0

    for filename in sorted(os.listdir(dir_path)):
        if count_loaded >= max_images:
            break

        file_path = os.path.join(dir_path, filename)
        with np.load(file_path) as data:
            for key in data.files:
                if count_loaded >= max_images:
                    break

                matrix = data[key]
                num_rows, num_cols = matrix.shape

                if 1 <= num_rows <= max_rows:

                    max_vals = np.max(matrix, axis=0)
                    normalized_matrix = np.divide(
                        matrix,
                        max_vals,
                        out=np.zeros_like(matrix),  # Where max_vals == 0
                        where=max_vals != 0
                    )

                    pad_width = ((0, max_rows - num_rows), (0, 0))
                    padded_matrix = np.pad(normalized_matrix, pad_width, mode='constant')

                    image_array = (padded_matrix * 255).astype(np.uint8)
                    images.append(image_array)

                    count_loaded += 1

    if len(images) == 0:
        raise ValueError(f"No valid matrices found in {dir_path}.")

    images = np.array(images)

    images = np.expand_dims(images, axis=-1)
    return images


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

image_sets = [load_npz(d) for d in dir_paths]
for i, imgs in enumerate(image_sets):
    print(f"Images {i+1} shape: {imgs.shape}")

all_images = np.concatenate(image_sets, axis=0)

all_labels = []
for class_index, imgs in enumerate(image_sets):
    labels_for_this_class = np.full(len(imgs), class_index)
    all_labels.append(labels_for_this_class)
all_labels = np.concatenate(all_labels)

num_classes = len(dir_paths)
all_labels = to_categorical(all_labels, num_classes=num_classes)

print("All images shape:", all_images.shape)
print("All labels shape:", all_labels.shape)

#################################################################################
################# TRAIN/TEST SPLIT ##############################################
#################################################################################

X_train, X_test, y_train, y_test = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#################################################################################
################# MODEL #########################################################
#################################################################################

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(100, 7, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Flatten(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#################################################################################
################# TRAINING ######################################################
#################################################################################

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
]
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

#################################################################################
################# EVALUATION ####################################################
#################################################################################

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

#################################################################################
############# PLOTTING ##########################################################
#################################################################################

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()
