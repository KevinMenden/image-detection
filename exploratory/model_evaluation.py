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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, utils


class ImageDataset(Dataset):
    """Test Class"""

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

data_path = "/home/kevin/deep_learning/OpenImages/"
eval_freq = 50

writer = SummaryWriter(log_dir=os.path.join(data_path, "models"))

transform = transforms.Compose(
    [transforms.Scale((299, 299)),
     transforms.Grayscale(3),
     transforms.ToTensor()])

root_dir = os.path.join(data_path, "pics")
csv_path = os.path.join(data_path, "open_image_labels_formatted.csv")
label_name_path = os.path.join(data_path, "label_names.csv")
dataset = ImageDataset(label_file = csv_path, root_dir = root_dir,
                       label_name_path = label_name_path, transform=transform)

n_classes = len(dataset.label_names)
print(f"{n_classes} classes")

# Create model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)

# load state dict
model.load_state_dict(torch.load(os.path.join(data_path, "models", "resnet18")))

# specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

# Load a batch
batch = next(iter(data_loader))
labels = batch['label'].cpu().detach().numpy()
images = batch['image'].to(device)

# run through network
preds = model(images)

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

preds = preds.cpu().detach().numpy()

res = np.round(sigmoid(preds))

TP = np.sum(np.logical_and(res == 1, labels == 1))
TN = np.sum(np.logical_and(res == 0, labels == 0))
FP = np.sum(np.logical_and(res == 1, labels == 0))
FN = np.sum(np.logical_and(res == 0, labels == 1))

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TN + TP) / (TN + TP + FN + FP)
f1 = 2 * (sensitivity * specificity) / (sensitivity + specificity)