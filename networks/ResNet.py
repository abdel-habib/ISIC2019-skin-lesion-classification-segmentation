import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger

class ResNetMel(nn.Module):
    def __init__(self, num_classes, fine_tune=False, num_layers_to_unfreeze=None):
        """
        Initialize the ResNet model.

        Args:
            num_classes (int): Number of output classes.
            fine_tune (bool): Whether to fine-tune the model.
            num_layers_to_unfreeze (int): Number of layers to unfreeze during fine-tuning.
        """
        super(ResNetMel, self).__init__()

        logger.info(f"Using ResNetMel with configurations: num_classes='{num_classes}', fine_tune='{fine_tune}', num_layers_to_unfreeze='{num_layers_to_unfreeze}'")

        # Load pre-trained ResNet-50 model
        self.model = models.resnet50(pretrained=True)

        if fine_tune:
            # Fine-tune: Unfreeze specified number of layers in layer4
            if num_layers_to_unfreeze is not None:
                for param in self.model.layer4[-num_layers_to_unfreeze:].parameters():
                    param.requires_grad = True
            else:
                # Fine-tune: Unfreeze all layers in layer4
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
        else:
            # Transfer Learning: Freeze all layers except the final layer
            for param in self.model.parameters():
                param.requires_grad = False

        # Modify the last fully connected layer for the number of classes in your problem
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # only perform this if transfer learning is used
        if not fine_tune:
            self.model.fc.requires_grad = True

        # Add dropout for regularization
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.model(x)
        # x = self.dropout(x)  # Apply dropout after the pre-trained layers

        return x

class SEResnext50_32x4d(nn.Module):
    '''SE ResNeXt is a variant of a ResNext that employs squeeze-and-excitation blocks to enable the network to perform dynamic 
    channel-wise feature recalibration.
    
    https://paperswithcode.com/model/seresnext?variant=seresnext50-32x4d
    '''
    def __init__(self, num_classes, pretrained=True):
        super(SEResnext50_32x4d, self).__init__()

        logger.info(f"Using SEResnext50_32x4d with configurations: num_classes='{num_classes}', pretrained='{pretrained}'")

        # Load pre-trained ResNeXt-50 32x4d model
        self.base_model = models.resnext50_32x4d(pretrained=pretrained)

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        # Add an adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Add the final fully connected layer
        in_features = self.base_model.fc.in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x