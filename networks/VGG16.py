import torch
import torch.nn as nn
import torchvision.models as models

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
