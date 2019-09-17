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
from torchvision import transforms, utils


class ImageDataset(Dataset):
    """Test Class"""

    def __init__(self, label_file, root_dir, label_name_path, transform=None):
        self.labels = pd.read_csv(label_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.label_names = pd.read_csv(label_name_path, index_col=0)

        self.images = list(set(self.labels.index))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """ Get an image"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        img_id = self.images[idx]
        img_path = os.path.join(self.root_dir, self.images[idx] + ".jpg")
        image = Image.open(img_path)

        # Create label vector
        image_label = np.array(self.labels.loc[img_id].item().split(",")).astype(int)

        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        image = np.asarray(image)
        sample = {'image': image, 'label': image_label}

        return sample


data_path = "/run/media/kevin/Volume/OpenImages/"

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

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)

# specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

data_loader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)
len(data_loader)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, batch in enumerate(data_loader):
        # get the inputs; data is a list of [inputs, labels]
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(imgs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Save the model
torch.save(model.state_dict(), os.path.join(data_path, "models", "resnet18"))