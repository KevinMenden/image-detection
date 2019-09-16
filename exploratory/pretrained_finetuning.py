"""
Image detection using some pre-trained model and fine tuning on
the a subset of the OpenImage database
Mainly to get the input pipeline up and running
"""
import torch
import os
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image


class ImageDataset(Dataset):
    """
    ImageDataset class for loading images and creating label vectors
    """

    def __init__(self, label_file, root_dir, transform=None):
        self.labels = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform

        # Make some adjustments
        self.labels.index = self.labels['ImageID']
        self.labels = self.labels[self.labels['Confidence'] == 1]
        self.label_names = np.sort(list(set(self.labels['LabelName'])))
        self.images = list(set(self.labels['ImageID']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Get an image with labels"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_path = os.path.join(self.root_dir, self.images[idx] + ".jpg")
        image = np.asarray(Image.open(img_path))

        # Create label vector
        img_id = self.images[idx]
        img_objects = np.array(self.labels.loc[img_id, ]['LabelName'])
        image_label = np.array([1 if x in img_objects else 0 for x in self.label_names])

        # Return sample
        sample = {'image': image, 'label': image_label}
        return sample

# Parameters
n_classes = 601

# Load the resnet18 model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
