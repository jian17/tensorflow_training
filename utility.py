import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                    if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        filenames = [os.path.join(label_directory, f)
                     for f in os.listdir(label_directory)
                     if f.endswith(".png")]
        for f in filenames:
            images.append(skimage.io.imread(f))
            labels.append(int(d))

    return images, labels

def show_random_image(images, labels):
    unique_labels = set(labels)

    plt.figure(figsize=(15, 15))
    i = 1

    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
        
    plt.show()

def preprocess_data(images, size=32):
    images_resized = [skimage.transform.resize(image, (size, size)) for image in images]
    #images_gray = [skimage.color.rgb2gray(image) for image in images_resized]
    return np.array(images_resized)

def get_default_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(32,32)))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(62, activation="softmax"))
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model
