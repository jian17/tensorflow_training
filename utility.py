import os
import skimage
import numpy as np

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

    return np.array(images), np.array(labels)