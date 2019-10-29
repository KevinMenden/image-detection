"""
Functions and classes for data loading and processing
"""
"""
Image detection using some pre-trained model and fine tuning on
the a subset of the OpenImage database
Mainly to get the input pipeline up and running
"""
import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, utils


class WildcamDataset(Dataset):
    """Wildcam Image Dataset"""

    def __init__(self, label_file, root_dir, transform=None, n_classes=23):
        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform
        self.n_classes = n_classes
        # Get images
        self.images = np.array(self.labels['file_name'])

        # Create one-hot encoded labels
        lab_vec = np.zeros((self.labels.shape[0], self.n_classes))
        for idx in range(lab_vec.shape[0]):
            lab_vec[idx][self.labels['category_id'][idx]] = 1
        self.labels = lab_vec



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Get an image"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, img))

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return {'image': image, 'label': self.labels[idx]}


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def calc_metrics(predictions, targets):
    predictions = np.round(sigmoid(predictions))

    TP = np.sum(np.logical_and(predictions == 1, targets == 1))
    TN = np.sum(np.logical_and(predictions == 0, targets == 0))
    FP = np.sum(np.logical_and(predictions == 1, targets == 0))
    FN = np.sum(np.logical_and(predictions == 0, targets == 1))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TN + TP) / (TN + TP + FN + FP)
    f1 = 2 * (sensitivity * specificity) / (sensitivity + specificity)
    return (f1, accuracy, sensitivity, specificity)
