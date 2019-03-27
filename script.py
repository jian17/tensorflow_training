import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from utility import *

ROOT_PATH = os.getcwd()
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

images, labels = load_data(train_data_directory)
plt.hist(labels, 62)
plt.show()
