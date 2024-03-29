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

writer = SummaryWriter(log_dir=os.path.join(data_path, "models"))

transform = transforms.Compose(
    [transforms.Scale((299, 299)),
     transforms.Grayscale(3),
     transforms.ToTensor()])

root_dir = os.path.join(data_path, "pics")
csv_path = os.path.join(data_path, "open_image_labels_formatted.csv")
label_name_path = os.path.join(data_path, "label_names.csv")

# Create dataset and train-test split
dataset = ImageDataset(label_file = csv_path, root_dir = root_dir,
                       label_name_path = label_name_path, transform=transform)

train_size = int(0.98 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

# Get number of classes
n_classes = len(dataset.label_names)
print(f"{n_classes} classes")

# Create model
model = models.resnet18(pretrained=False)
test = models.resnet18(pretrained=False)
model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)

# load state dict
#model.load_state_dict(torch.load(os.path.join(data_path, "models", "resnet18")))

# specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Loss and optimzier
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())



eval_freq = 100
test_freq = 1000
global_step = 0
for epoch in range(10, 20):  # loop over the dataset multiple times

    running_step = 0
    for i, batch in enumerate(train_loader):
        model.train()
        running_step += 1
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

        # Print training performance
        current_loss = loss.item()
        global_step += 1
        if i % eval_freq == 0:
            train_f1, train_accuracy, sensitivity, specificity = calc_metrics(outputs.cpu().detach().numpy(), labels.cpu().detach().numpy())
            print(f"[{epoch} - {global_step}]: {current_loss}, F1: {train_f1}, Acc: {train_accuracy}")


        # Try to free up GPU memory
        del imgs
        del labels
        torch.cuda.empty_cache()

        # Evaluate on hold-out data

        if i % test_freq == 0:
            running_loss = 0
            running_f1 = 0
            running_accuracy = 0
            model.eval()
            for j, batch in enumerate(test_loader):
                imgs = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels.float())
                f1, accuracy, _, _ = calc_metrics(outputs.cpu().detach().numpy(),
                                                                      labels.cpu().detach().numpy())
                running_loss += loss.item()
                running_f1 += f1
                running_accuracy += accuracy

            running_loss = running_loss / (j+1)
            running_f1 = running_f1 / (j+1)
            running_accuracy = running_accuracy / (j+1)
            print(f"Eval: {running_loss}, {running_f1}, {running_accuracy}")

            # Add scalars
            writer.add_scalars('Loss', {'Train': current_loss,
                                        'Eval': loss.item()}, global_step)
            writer.add_scalars('F1', {'Train': train_f1,
                                      'Eval': f1}, global_step)
            writer.add_scalars('Accuracy', {'Train': train_accuracy,
                                            'Eval': accuracy}, global_step)

    torch.save(model.state_dict(), os.path.join(data_path, "models", "resnet18"))

print('Finished Training')

# Save the model