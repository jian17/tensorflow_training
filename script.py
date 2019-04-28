#%%
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from utility import (load_data, preprocess_data, get_default_model)
import random
import skimage
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
model = get_default_model()

#%%
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

#%%
model.save_weights('my_model.h5', save_format="h5")

#%%
model.load_weights('my_model.h5')

#%%
predictions = model.predict(test_images)

#%%
indexes = []
for i in range(16):
    indexes.append(random.randint(0, len(test_images)))

plt.figure(figsize=(15,15))
j = 1
for i in indexes:
    image = test_images[i]
    plt.subplot(4,4,j)
    plt.axis("off")
    plt.set_cmap("gray")
    pred_index = np.argmax(predictions[i])
    pred_acc = predictions[i][pred_index] * 100
    plt.title("{0}: {1} ({2:.2f})".format(pred_index, test_labels[i], pred_acc))
    j += 1
    plt.imshow(image)
plt.show()