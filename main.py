"""
Image detection using some pre-trained model and fine tuning on
the a subset of the OpenImage database
Mainly to get the input pipeline up and running
"""
import torch
import os
from torch.utils.data import  DataLoader, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from archive.utils import WildcamDataset, calc_metrics
from torchvision import models
from efficientnet_pytorch import EfficientNet

# Define paths, parameters and variables
data_path = "/home/kevin/deep_learning/iwildcam"
root_dir = os.path.join(data_path, "train_images")
label_file = os.path.join(data_path, "train.csv")


# Image transform
transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomPerspective(),
     transforms.RandomResizedCrop((299, 299)),
     transforms.RandomRotation(30),
     transforms.Grayscale(3),
     transforms.ToTensor()])

# Create a summary writer
writer = SummaryWriter(log_dir=os.path.join(data_path, "models/effnet0/"))



# Create dataset and train-test split
dataset = WildcamDataset(label_file=label_file, root_dir=root_dir, transform=transform)

data_size = len(dataset)
test_size = 100
train_size = data_size - test_size
train_dataset = Subset(dataset, list(range(0, train_size)))
test_dataset = Subset(dataset, list(range(train_size, data_size)))

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=6, pin_memory=True)

# Get number of classes
n_classes = dataset.n_classes
print(f"{n_classes} classes")

# Create model
#model = ResNet(n_classes=n_classes, in_size=64)
model = EfficientNet.from_pretrained("efficientnet-b0")
#model = models.resnet34(pretrained=False)
model._fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)

# load params
#model.load_state_dict(torch.load(os.path.join(data_path, "models", "resnet50/resnet_model")))

# specify device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Loss and optimzier
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

eval_freq = 100
test_freq = 1000
global_step = 0
for epoch in range(0, 15):  # loop over the dataset multiple times

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

    torch.save(model.state_dict(), os.path.join(data_path, "models", "effnet0/resnet_model"))

print('Finished Training')