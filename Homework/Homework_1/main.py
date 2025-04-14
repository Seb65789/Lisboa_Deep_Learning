import torch
import numpy as np
import matplotlib.pyplot as plt

# Importing the classes
#================================================================================================#

from src.linearmodels import Perceptron


# Load the dataset
#================================================================================================#
print("Loading dataset...")
data = np.load("./Homework/Homework_1/src/intel_landscapes.npz")

X_train = data['train_images']
Y_train = data['train_labels']

X_val = data['val_images']
Y_val = data['val_labels']

X_test = data['test_images']
Y_test = data['test_labels']

plt.imshow(X_train[0])
plt.title(f"Label: {Y_train[0]}")
plt.savefig("./Homework/Homework_1/test.png")

#================================================================================================#