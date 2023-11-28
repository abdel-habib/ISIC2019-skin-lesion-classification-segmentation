import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from loguru import logger

from torchsummary import summary

class BilinearInterpolationBlock(nn.Module):
    def __init__(self, scale_factor):
        super(BilinearInterpolationBlock, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

class AttentionBlock(nn.Module):
    '''
    A modified implementation for 'Melanoma Recognition via Visual Attention'. Original work can be found in the published work by 
    the author. 

    @inproceedings{yan2019melanoma,
        title={Melanoma Recognition via Visual Attention},
        author={Yan, Yiqi and Kawahara, Jeremy and Hamarneh, Ghassan},
        booktitle={International Conference on Information Processing in Medical Imaging},
        pages={793--804},
        year={2019},
        organization={Springer}
        }
    '''
    def __init__(self, in_features_F, in_features_G, out_features, scale_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        
        # convolutional blocks
        self.conv_blockF = nn.Conv2d(in_channels=in_features_F, out_channels=out_features, kernel_size=1, padding=0, bias=False)
        self.conv_blockG = nn.Conv2d(in_channels=in_features_G, out_channels=out_features, kernel_size=1, padding=0, bias=False)
        
        # attention map block, set bias to True to adds a learnable bias to theoutput
        self.conv_attenM = nn.Conv2d(in_channels=out_features, out_channels=1, kernel_size=1, padding=0, bias=True)

        # biliniar interpolation to a given scale factor
        self.intepolator = BilinearInterpolationBlock(scale_factor)

        # Relu
        self.relu = nn.ReLU()

        # attention map layer
        self.attention_map_sigmoid = nn.Sigmoid()
        self.attention_map_softmax = nn.Softmax()

        # check if the attention map should be normalized
        self.normalize_attn = normalize_attn

    def forward(self, _F, _G):
        N, C, W, H = _F.size()  # Extract N, C, W, H from the input tensor F

        # conv blocks
        xF = self.conv_blockF(_F)
        xG = self.conv_blockG(_G)

        # biliniar interpolation
        xG = self.intepolator(xG)

        # element-wise sum operation with relu
        xAttention = self.relu(xF + xG)

        # pixel-wise multiplication using convolution
        responseR = self.conv_attenM(xAttention) # torch.Size([2, 1, 28, 28])

        # check if the attention map should be normalized
        if self.normalize_attn:    
            # Apply softmax along the spatial dimensions
            # A = self.attention_map_softmax(responseR.view(N, 1, -1), dim=2).view(N, 1, H, W)
            A = F.softmax(responseR.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            # Apply sigmoid along the spatial dimensions
            A = self.attention_map_sigmoid(responseR)

        # Re-weight the local features
        f = torch.mul(A.expand_as(_F), _F)  # batch_size x channels x W x H
        
        if self.normalize_attn:    
            # Weighted sum
            output = f.view(N, C, -1).sum(dim=2)
        else:
            # Global average pooling
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)

        return A, output

class VGG16_BN_Attention(nn.Module):
    '''
    A modified implementation for 'Melanoma Recognition via Visual Attention'. Original work can be found in the published work by 
    the author. 

    @inproceedings{yan2019melanoma,
        title={Melanoma Recognition via Visual Attention},
        author={Yan, Yiqi and Kawahara, Jeremy and Hamarneh, Ghassan},
        booktitle={International Conference on Information Processing in Medical Imaging},
        pages={793--804},
        year={2019},
        organization={Springer}
        }
    '''
    def __init__(self, num_classes, normalize_attn=False):
        super(VGG16_BN_Attention, self).__init__()
        logger.info(f"Using VGG16_BN_Attention with configurations: num_classes='{num_classes}', normalize_attn='{normalize_attn}'")

        # load vgg16_bn pre-trained model
        vgg16_bn = models.vgg16_bn(pretrained=True)

        # select each block separately to integrate the attention blocks to the network
        # Note: we select the conv blocks from the features layers without the first max-pooling layers of each block 
        # [6, 13, 23, 33, 43]  
        self.conv_block1 = nn.Sequential(*list(vgg16_bn.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(vgg16_bn.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(vgg16_bn.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(vgg16_bn.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(vgg16_bn.features.children())[34:43])

        # attention blocks
        self.attention_block1 = AttentionBlock(256, 512, 256, 4, normalize_attn)
        self.attention_block2 = AttentionBlock(512, 512, 256, 2, normalize_attn)

        # create the remaining layers of the model
        self.avgpool2d = nn.AvgPool2d(7, stride=1)        
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # select the classifier without the final fully connected layer
        self.classifier  = nn.Linear(in_features=512+512+256, out_features=num_classes, bias=True)

        # initialize the weights
        self.initialize_weights(self.classifier)
        self.initialize_weights(self.attention_block1)
        self.initialize_weights(self.attention_block2)

    def initialize_weights(self, module, method='kaiming_normal'):
        if method == 'kaiming_normal':
            # initialize the weights using the kaiming_normal method
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.)
                    nn.init.constant_(m.bias, 0.)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0., 0.01)
                    nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        # block 1
        x = self.conv_block1(x)
        pool1 = self.maxpool2d(x)

        # block 2
        x = self.conv_block2(pool1)
        pool2 = self.maxpool2d(x)

        # block 3
        x = self.conv_block3(pool2)
        pool3 = self.maxpool2d(x)

        # block 4
        x = self.conv_block4(pool3)
        pool4 = self.maxpool2d(x)

        # block 5
        x = self.conv_block5(pool4)
        pool5 = self.maxpool2d(x)

        N, C, __, __ = pool5.size()

        # # AdaptiveAvgPool2d
        x = self.avgpool2d(pool5)
        
        # reshape the tensor to make it possible for concatenation
        x = x.view(N, C, -1)  # flatten the tensor
        x = x.mean(dim=-1)  # take the mean along the spatial dimensions

        # # attention blocks
        att1, op1 = self.attention_block1(pool3, pool5)
        att2, op2 = self.attention_block2(pool4, pool5)
        # print(op1.shape, op2.shape, x.shape)

        # concatinate the features
        x = torch.cat((x, op1, op2), dim=1)
        # print(x.shape)

        # classifier block
        x = self.classifier(x)
        # print(x.shape)

        return x


class VGG16_BN(nn.Module):
    '''
    A modified implementation for 'Melanoma Recognition via Visual Attention'. Original work can be found in the published work by 
    the author. 

    @inproceedings{yan2019melanoma,
        title={Melanoma Recognition via Visual Attention},
        author={Yan, Yiqi and Kawahara, Jeremy and Hamarneh, Ghassan},
        booktitle={International Conference on Information Processing in Medical Imaging},
        pages={793--804},
        year={2019},
        organization={Springer}
        }
    '''
    def __init__(self, num_classes):
        super(VGG16_BN, self).__init__()
        logger.info(f"Using VGG16_BN with configurations: num_classes='{num_classes}'")

        # load vgg16_bn pre-trained model
        vgg16_bn = models.vgg16_bn(pretrained=True)

        # select each block separately to integrate the attention blocks to the network
        # Note: we select the conv blocks from the features layers without the first max-pooling layers of each block 
        # [6, 13, 23, 33, 43]  
        self.conv_block1 = nn.Sequential(*list(vgg16_bn.features.children())[0:6])
        self.conv_block2 = nn.Sequential(*list(vgg16_bn.features.children())[7:13])
        self.conv_block3 = nn.Sequential(*list(vgg16_bn.features.children())[14:23])
        self.conv_block4 = nn.Sequential(*list(vgg16_bn.features.children())[24:33])
        self.conv_block5 = nn.Sequential(*list(vgg16_bn.features.children())[34:43])

        # create the remaining layers of the model
        self.avgpool2d = nn.AvgPool2d(7, stride=1)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # select the classifier without the final fully connected layer
        self.classifier  = nn.Sequential(*list(vgg16_bn.classifier.children()))

        # create the final fully connected layer that matches the output of our problem
        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, num_classes)

        # set requires_grad to False for all layers except the modified layer
        for param in self.parameters():
            param.requires_grad = False

        # set requires_grad to True for the modified layer
        for param in self.classifier[-1].parameters():
            param.requires_grad = True

    def forward(self, x):
        # block 1
        x = self.conv_block1(x)
        pool1 = self.maxpool2d(x)

        # block 2
        x = self.conv_block2(pool1)
        pool2 = self.maxpool2d(x)

        # block 3
        x = self.conv_block3(pool2)
        pool3 = self.maxpool2d(x)

        # block 4
        x = self.conv_block4(pool3)
        pool4 = self.maxpool2d(x)

        # block 5
        x = self.conv_block5(pool4)
        pool5 = self.maxpool2d(x)

        # AdaptiveAvgPool2d
        x = self.avgpool2d(pool5)
        
        # flatten the 3D tensor into a 1D tensor, 
        # (batch_size, -1), where -1 infers the size based on the remaining dimensions.
        # from torch.Size([2, 512, 7, 7]) to torch.Size([2, 25088])
        x = x.view(x.size(0), -1)

        # classifier bock
        x = self.classifier(x)

        return x