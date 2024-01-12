import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

def visualize_attn(I, a, up_factor, nrow):
    '''
    Visualizes attention maps. Implementation from: 
        https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/utilities.py

    Args:
        I (torch.Tensor): Input image batch.
        a (torch.Tensor): Attention map batch.
        up_factor (int): Upsampling factor.
        nrow (int): Number of rows in the grid.

    Returns:
        torch.Tensor: Visualized attention map.
    '''
    
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)

    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255

    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)


