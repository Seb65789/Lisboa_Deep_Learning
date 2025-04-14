import matplotlib as plt
import numpy as np

# Load the dataset
#================================================================================================#

def load_dataset(path):
    print("Loading dataset...")
    data = np.load(path)

    X_train = data['train_images']
    Y_train = data['train_labels']

    X_val = data['val_images']
    Y_val = data['val_labels']

    X_test = data['test_images']
    Y_test = data['test_labels']

    plt.imshow(X_train[0])
    plt.title(f"Label: {Y_train[0]}")
    plt.savefig("./Homework/Homework_1/test.png")

    return X_train,X_val,X_test,Y_train,Y_val,Y_test

#================================================================================================#