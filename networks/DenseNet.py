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





# class DenseNet(nn.Module):
#     def __init__(self, num_classes, fine_tune=False, num_layers_to_unfreeze=None):
#         super(DenseNet, self).__init__()

#         # Load pre-trained DenseNet-121 model
#         self.model = models.densenet121(pretrained=True)

#         # Freeze or unfreeze layers based on fine-tune setting
#         if fine_tune:
#             if num_layers_to_unfreeze is not None:
#                 # Unfreeze the specified number of layers
#                 for param in self.model.features.denseblock4[-num_layers_to_unfreeze:].parameters():
#                     param.requires_grad = True
#             else:
#                 # Fine-tune: Unfreeze all layers
#                 for param in self.model.parameters():
#                     param.requires_grad = True
#         else:
#             # Transfer Learning: Freeze all layers except the final classifier layer
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             self.model.classifier.requires_grad = True

#         # Modify the architecture
#         # First convolutional layer
#         self.model.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # Add dense blocks and transition layers
#         self.model.features.denseblock1 = self._make_dense_block(6, 64)
#         self.model.features.transition1 = self._make_transition(64)

#         self.model.features.denseblock2 = self._make_dense_block(12, 128)
#         self.model.features.transition2 = self._make_transition(128)

#         self.model.features.denseblock3 = self._make_dense_block(24, 256)
#         self.model.features.transition3 = self._make_transition(256)

#         self.model.features.denseblock4 = self._make_dense_block(16, 512)

#         # Average pooling layer
#         self.avg_pool_7x7 = nn.AvgPool2d(kernel_size=7, stride=2, padding=3)

#         # Final fully connected layer
#         in_features = self.model.classifier.in_features
#         self.model.classifier = nn.Sequential(
#             self.avg_pool_7x7,
#             nn.Flatten(),
#             nn.Linear(in_features, num_classes)
#         )

#     def _make_dense_block(self, num_layers, num_input_features):
#         layers = []
#         for i in range(num_layers):
#             layers.append(models.densenet._DenseLayer(num_input_features + i * self.model.features.growth_rate, self.model.features.growth_rate))
#         return nn.Sequential(*layers)

#     def _make_transition(self, num_output_features):
#         return models.densenet._Transition(self.model.features.num_features, num_output_features)

#     def forward(self, x):
#         """
#         Defines the forward pass of the Modified DenseNet model.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             torch.Tensor: Output tensor.
#         """
#         return self.model(x)
    
# '''
# https://doi.org/10.1016/j.cmpb.2021.106174
# '''

# class DenseNet121_v145(nn.Module):
#     def __init__(
#             self, 
#             num_init_features: int = 64, 
#             block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
#             num_classes=None):
        
#         super(DenseNet121_v145, self).__init__()

#         # First convolution from DenseNet package
#         self.features = nn.Sequential(
#             OrderedDict(
#                 [
#                     ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#                     ("norm0", nn.BatchNorm2d(num_init_features)),
#                     ("relu0", nn.ReLU(inplace=True)),
#                     ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#                 ]
#             )
#         )

#         # Building each dense/transition block
#         num_features = num_init_features

# class BottleneckLayer(nn.Module):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(BottleneckLayer, self).__init__()

#         # 1x1 Convolution (Bottleneck layer)
#         self.norm1 = nn.BatchNorm2d(num_input_features)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

#         # 3x3 Convolution
#         self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

#         self.drop_rate = float(drop_rate)

#     def forward(self, x):
#         # Bottleneck layer
#         bottleneck_output = self.conv1(self.relu1(self.norm1(x)))
        
#         # 3x3 Convolution
#         out = self.conv2(self.relu2(self.norm2(bottleneck_output)))

#         # Apply dropout if specified
#         if self.drop_rate > 0:
#             out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)

#         # Concatenate the input and output along the channel dimension
#         return torch.cat([x, out], 1)


# class DenseBlock(nn.Module):
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
#         super(DenseBlock, self).__init__()
#         layers = []
#         for i in range(num_layers):
#             layers.append(BottleneckLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))

#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.layers(x)


# if __name__ == "__main__":
#     # DenseNet121_Mel()   
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     weights_default = models.DenseNet121_Weights.DEFAULT

#     pretrained_densenet121 = models.densenet121(weights=weights_default)
#     print(pretrained_densenet121)

#     # summary(
#     #     pretrained_densenet121.to(device), input_size=(3, 224, 224)) 

#     # dense_block = DenseBlock(num_layers=6, 
#     #                          num_input_features = 64, 
#     #                          bn_size = 4, 
#     #                          growth_rate = 32, 
#     #                          drop_rate=0)
    
#     # print(dense_block)

