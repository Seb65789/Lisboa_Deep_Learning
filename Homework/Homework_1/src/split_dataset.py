import numpy as np
import matplotlib.pyplot as plt

print("Loading dataset...")

# Load the dataset
data = np.load("./Homework/Homework_1/src/intel_landscapes.npz")
print(data)

X_train = data['train_images']
Y_train = data['train_labels']

X_val = data['val_images']
Y_val = data['val_labels']

X_test = data['test_images']
Y_test = data['test_labels']

plt.imshow(X_train[1]) 
plt.savefig('./Homework/Homework_1/src/test_image.png')

print(Y_train[1])

print("Number of train images :",len(X_train))
print("Number of validation images :", len(X_val))
print("Number of test images :", len(X_test))