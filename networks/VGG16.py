import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchsummary import summary

class VGG16_BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_BN, self).__init__()
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
        self.avgpool2d = nn.AdaptiveAvgPool2d(output_size=(7, 7))        
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
        x = self.maxpool2d(x)

        # block 2
        x = self.conv_block2(x)
        x = self.maxpool2d(x)

        # block 3
        x = self.conv_block3(x)
        x = self.maxpool2d(x)

        # block 4
        x = self.conv_block4(x)
        x = self.maxpool2d(x)

        # block 5
        x = self.conv_block5(x)
        x = self.maxpool2d(x)

        # AdaptiveAvgPool2d
        x = self.avgpool2d(x)
        
        # flatten the 3D tensor into a 1D tensor, 
        # (batch_size, -1), where -1 infers the size based on the remaining dimensions.
        # from torch.Size([2, 512, 7, 7]) to torch.Size([2, 25088])
        x = x.view(x.size(0), -1)

        # classifier bock
        x = self.classifier(x)

        return x








class VGG16Transfer(nn.Module):
    def __init__(self):
        super(VGG16Transfer, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class VGG16Mel(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Mel, self).__init__()
        self.vgg_transfer = VGG16Transfer()

        # Freezing the first layers
        # for param in self.vgg_transfer.parameters():
        #     param.requires_grad = False

        # Freeze layers up to the 17th layer (exclusive)
        for i, layer in enumerate(self.vgg_transfer.features):
            if i < 17:
                for param in layer.parameters():
                    param.requires_grad = False
        

        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.vgg_transfer(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    # Instantiate the model
    custom_model = VGG16Mel(num_classes=2)
    print(custom_model)
