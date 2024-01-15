import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetMel(nn.Module):
    def __init__(self, weights_b=1, num_classes=2, fine_tune=False, num_layers_to_unfreeze=None): 
        super(EfficientNetMel, self).__init__()
        
        self.weights = [
            'efficientnet-b0',
            'efficientnet-b1',
            'efficientnet-b2',
            'efficientnet-b3',
            'efficientnet-b4',
            'efficientnet-b5',
            'efficientnet-b6',
            'efficientnet-b7'
        ]

        # load a pretrained model with a new number of classes for transfer learning
        self.model = EfficientNet.from_pretrained(self.weights[weights_b], num_classes=num_classes)
        
        # Freeze layers if specified
        if fine_tune:
            # Determine the total number of layers in the model
            total_layers = len(list(self.model.parameters()))
            
            # Specify the number of layers to keep trainable from the end
            num_layers_to_train = num_layers_to_unfreeze
            
            # Determine the starting index to freeze layers
            start_index_to_freeze = total_layers - num_layers_to_train
            
            # Freeze layers except the last num_layers_to_train layers
            for i, param in enumerate(self.model.parameters()):
                if i < start_index_to_freeze:
                    param.requires_grad = False

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
    