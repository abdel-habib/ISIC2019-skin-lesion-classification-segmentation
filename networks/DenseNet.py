'''Paper: Multi-Class Breast Cancer Classification using Deep
Learning Convolutional Neural Network'''

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
from torchsummary import summary

class DenseNetMel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetMel, self).__init__()

        # Load pre-trained DenseNet model
        densenet_model = models.densenet161(pretrained=True)

        # Transfer Learning: Freeze all layers except the final layer
        for param in densenet_model.parameters():
            param.requires_grad = False

        self.features = densenet_model.features  # Extracting the feature extractor part of DenseNet
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer
        self.fc = nn.Linear(densenet_model.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



class DenseNetMel0(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetMel, self).__init__()

        # Load pre-trained DenseNet model
        densenet_model = models.densenet161(pretrained=True)

        # Transfer Learning: Freeze all layers except the final layer
        for param in densenet_model.parameters():
            param.requires_grad = False

        # Remove the original fully connected layer (classifier in DenseNet is a fully connected layer)
        self.features = densenet_model.features
        self.in_features = densenet_model.classifier.in_features

        # Custom dense blocks and transition layers
        self.dense_blocks = nn.ModuleList([self._make_dense_block(in_channels=self.in_features, growth_rate=32, num_layers=6) for _ in range(4)])
        self.transition_layers = nn.ModuleList([self._make_transition_layer(in_channels=self.in_features) for _ in range(4)])

        # Adjust the input size based on the DenseNet architecture
        self.fc = nn.Linear(self.in_features, num_classes)  # The input size of fc layer should match the output size before global avg pooling

    def forward(self, x):
        # Feature extraction using DenseNet
        x = self.features(x)

        # Custom dense blocks and transition layers
        for i in range(len(self.dense_blocks)):
            x = self.dense_blocks[i](x)
            x = self.transition_layers[i](x)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten before the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )