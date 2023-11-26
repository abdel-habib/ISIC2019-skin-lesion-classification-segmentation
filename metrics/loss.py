import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        '''
        Implementation of focal loss function to address class imbalance problem. Focal loss applies a modulating term to the cross entropy loss 
        in order to focus learning on hard misclassified examples. It is a dynamically scaled cross entropy loss, where the scaling factor decays 
        to zero as confidence in the correct class increases. 

        Args:
            gamma (float): Modulating factor to adjust the rate at which the scaling factor decays. Default: 2
            weight (tensor): Weighting factor to adjust the rate at which the scaling factor decays. Default: None
            reduction (string): Reduction option to apply to the loss. Default: 'mean'

        References:
            Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. 
            In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).

        Returns:
            loss (float): Focal loss value.
        '''
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # compute the cross entropy loss
        ce_loss = -F.cross_entropy(input, target, weight=self.weight, reduction='none')

        # compute the sigmoid term
        pt = torch.exp(ce_loss)

        # compute the modulating term
        modulating_factor = -(1 - pt) ** self.gamma

        # compute the focal loss
        focal_loss = modulating_factor * ce_loss

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
    