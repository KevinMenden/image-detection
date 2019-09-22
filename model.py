"""
Model for image detection
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


class ResNetBlock(nn.Module):
    """
    Block of the ResNet model
    """
    def __init__(self, in_size, out_size, stride_conv1=1, stride_conv2=1):
        super(ResNetBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.stride_conv1 = stride_conv1
        # Define layers of ResNetBlock
        self.conv1 = nn.Conv2d(in_size, out_size, stride=stride_conv1, kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, stride=stride_conv2, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)

        self.downsample = nn.Sequential(
            nn.Conv2d(self.in_size, self.out_size, stride_conv1),
            nn.BatchNorm2d(self.out_size)
        )

    def forward(self, x):
        # Save identity for residual function
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample the residual connection
        if self.in_size != self.out_size or self.stride_conv1 != 1:
            identity = self.downsample(identity)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Class for ResNet-like image classification model
    """

    def __init__(self, n_classes, in_size):
        super(ResNet, self).__init__()
        self.n_classes = n_classes
        self.in_size = in_size

        # Input layer of model
        self.conv1 = nn.Conv2d(3, self.in_size, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1
        self.layer1 = nn.Sequential(
            ResNetBlock(self.in_size, 64),
            ResNetBlock(self.in_size, 64)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            ResNetBlock(self.in_size, 128, stride_conv1=2, stride_conv2=1),
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            ResNetBlock(128, 128)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            ResNetBlock(128, 256, stride_conv1=2, stride_conv2=1),
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            ResNetBlock(256, 256)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock(256, 512, stride_conv1=2, stride_conv2=1),
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512),
            ResNetBlock(512, 512)
        )
        # Layer 4

        # head
        self.avpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, self.n_classes)

    def forward(self, x):
        # Input layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



model = ResNet(n_classes=601, in_size=64)