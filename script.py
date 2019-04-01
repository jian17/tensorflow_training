#%%
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from utility import (load_data, preprocess_data)

#%%
root_path = os.getcwd()
train_data_directory = os.path.join(root_path, "Training")
test_data_directory = os.path.join(root_path, "Testing")

#%%
train_images, train_labels = load_data(train_data_directory)
test_images, test_labels = load_data(test_data_directory)

#%%
train_images = preprocess_data(train_images)
test_images = preprocess_data(test_images)

#%%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(32,32)))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(62, activation="softmax"))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

#%%
model.save_weights('my_model.h5', save_format="h5")