import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                    if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        filenames = [os.path.join(label_directory, f)
                     for f in os.listdir(label_directory)
                     if f.endswith(".ppm")]
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