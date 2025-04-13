import numpy as np
import matplotlib.pyplot as plt

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

plt.imshow(X_train[1]) 
plt.savefig('./Homework/Homework_1/src/test_image.png')

print("Data Loaded")

print("Number of train images equal to the excepted number ? ",len(X_train)==12630)
print("Number of validation images equal to the excepted number ?", len(X_val)==1404)
print("Number of test images equal to the excepted number ?", len(X_test)==3000)

# Save the data
#================================================================================================#

print("Saving the data...")

np.save("./Homework/Homework_1/src/data/X_train.npz",X_train)
np.save("./Homework/Homework_1/src/data/X_val.npz",X_val)
np.save("./Homework/Homework_1/src/data/X_test.npz",X_test)

np.save("./Homework/Homework_1/src/data/Y_train.npz",Y_train)
np.save("./Homework/Homework_1/src/data/Y_val.npz",Y_val)
np.save("./Homework/Homework_1/src/data/Y_test.npz",Y_test)

print("Data saved.")

#================================================================================================#


