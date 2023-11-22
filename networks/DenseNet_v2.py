'''
https://doi.org/10.1016/j.cmpb.2021.106174

Implementation of DenseNet-145 network for extracting more abstract and deeper features.
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary

class _DenseBlockModified(nn.Module):
    def __init__(self, block_config, num_init_features=64, growth_rate=32, bn_size=4, drop_rate=0):
        super(_DenseBlockModified, self).__init__()

        for i, num_layers in enumerate(block_config):
            block = self._make_dense_layers(num_layers, num_init_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            setattr(self, f'denseblock{i + 1}', block)

    def _make_dense_layers(self, num_layers, in_channels, growth_rate, bn_size, drop_rate):
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate))
        return nn.Sequential(*layers)

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseNet145(nn.Module):
    def __init__(self, block_config, num_classes=2):
        super(DenseNet145, self).__init__()
        self.weights_default = models.DenseNet121_Weights.DEFAULT
        pretrained_densenet121 = models.densenet121(weights=self.weights_default)

        # Use the layers before the dense block from the pre-trained model
        self.features = nn.Sequential(*list(pretrained_densenet121.features.children())[:-1])

        # Modify the block configuration of the built-in _DenseBlock
        self.features[-1] = _DenseBlockModified(block_config)

        # Use the classifier from the pre-trained model
        self.classifier = pretrained_densenet121.classifier

        # Add the final batch normalization layer
        num_features = self._get_num_features(pretrained_densenet121)
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Add the linear layer
        self.final_layer = nn.Linear(num_features, num_classes)

    def _get_num_features(self, model):
        # Helper function to get the number of features before the classifier
        num_features = model.classifier.in_features
        return num_features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.final_layer(x)
        return x

if __name__ == "__main__":

    # Create an instance of the model
    densenet145 = DenseNet145(
        block_config=(6, 12, 24, 16)
    )

    # Print the model layers
    print(densenet145)
