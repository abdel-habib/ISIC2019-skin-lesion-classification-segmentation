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
