import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, weights=None, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        '''
        Implementation of the Focal Loss function to address class imbalance problems.

        Args:
            alpha (float): Weighting factor for the rare class. Default: 1
            gamma (float): Focusing parameter. Default: 2
            logits (bool): Whether the inputs are logits or not. Default: True
            weights (torch.Tensor): Weights for each class. Default: None
            reduction (str): Reduction option. Default: 'mean'

        Returns:
            loss (float): Focal loss value.
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        if not self.logits:
            # If logits=False, apply softmax to convert inputs to probabilities
            inputs = F.softmax(inputs, dim=1)

        # compute the cross entropy losses
        CE_loss = F.cross_entropy(inputs, targets, weight= self.weights, reduction='none')

        pt = torch.exp(-CE_loss)
        # focal_loss = - self.alpha * (1 - pt) ** self.gamma * CE_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss

        # check reduction option and return loss accordingly
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError('Unsupported reduction option.')


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        '''
        Implementation of the Dice Loss function to address class imbalance problems. 
        Dice Loss measures the similarity between the predicted binary mask and the target binary mask. 
        It is commonly used in image segmentation tasks to penalize the model for inaccurate pixel-wise predictions.

        Args:
            smooth (float): Smoothing factor to avoid division by zero. Default: 1

        Returns:
            loss (float): Dice loss value.
        '''
        self.smooth = smooth

    def forward(self, input, target):
        # compute the intersection
        intersection = torch.mul(input, target)

        # compute the addition
        addition = torch.add(input, target)

        # compute the dice loss
        dice_loss = torch.div((2 * intersection.sum() + self.smooth), (addition.sum() + self.smooth))

        return 1 - dice_loss
    