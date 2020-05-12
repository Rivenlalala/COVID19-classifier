import torch
import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):

    def __init__(self):

        super(DenseNet121, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=False)

        kernelCount = self.densenet121.classifier.in_features
        
        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

    
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16_bn(pretrained=False, num_classes=1)
        self.out = nn.Sequential(nn.BatchNorm1d(num_features=1),
                                  nn.Sigmoid())

    def forward(self, x):
        x = self.vgg16(x)
        x = self.out(x)
        return x
