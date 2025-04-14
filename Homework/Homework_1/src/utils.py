import os
import random

import numpy as np
import torch

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(data_path, bias=False):
    data = np.load(data_path)

    # Flatten images and normalize to get a grey image
    train_X = data["train_images"].reshape([data["train_images"].shape[0], -1])/256
    val_X = data["val_images"].reshape([data["val_images"].shape[0], -1])/256
    test_X = data["test_images"].reshape([data["test_images"].shape[0], -1])/256
   
    # flatten the labels with squeeze and stores as arrays
    train_y = np.asarray(data["train_labels"]).squeeze()
    val_y = np.asarray(data["val_labels"]).squeeze()
    test_y = np.asarray(data["test_labels"]).squeeze()

    # Add the bias for the mlp
    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        val_X = np.hstack((val_X, np.ones((val_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    # return an object
    return {
        "train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y),
    }

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        """
        data: the dict returned by utils.load_pneumonia_data
        """
        train_X, train_y = data["train"]
        val_X, val_y = data["val"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X, dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.val_X = torch.tensor(val_X, dtype=torch.float32)
        self.val_y = torch.tensor(val_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

