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


class ImageDataset(Dataset):
    """OpenImage Picture dataset"""

    def __init__(self, label_file, root_dir, label_name_path, transform=None):
        self.labels = pd.read_csv(label_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.label_names = pd.read_csv(label_name_path, index_col=0)

        # subset labels
        images = [x.replace(".jpg", "") for x in os.listdir(root_dir)]
        self.labels = self.labels.loc[images, ]

        self.images = list(self.labels.index)
        self.image_labels = [np.array(self.labels.loc[img_id].item().split(",")).astype(int) for img_id in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Get an image"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_id = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, img_id + ".jpg"))

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return {'image': image, 'label': self.image_labels[idx]}


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
