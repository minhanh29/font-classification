import torch
from torch import nn
import torch.nn.functional as F

class FontClassifier(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.refine = nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 1, padding = 1)
        self._conv1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1)
        self._conv1_bn = nn.BatchNorm2d(64)

        self._conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1)
        self._conv2_bn = nn.BatchNorm2d(64)

        self._conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1)
        self._conv3_bn = nn.BatchNorm2d(128)

        self._conv4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
        self._conv4_bn = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 512)
        self.out = nn.Tanh()

        self.classifier = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.refine(x))
        x = self._conv1(x)
        x = torch.nn.functional.leaky_relu(self._conv1_bn(x))
        x = self._conv2(x)
        x = torch.nn.functional.leaky_relu(self._conv2_bn(x))
        x = self._conv3(x)
        x = torch.nn.functional.leaky_relu(self._conv3_bn(x))
        x = self._conv4(x)
        x = torch.nn.functional.leaky_relu(self._conv4_bn(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)

        return x

    def forward_pair(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        dist = torch.abs(x1 - x2)
        out = self.classifier(dist)

        return out


class FontSiameseNet(nn.Module):
    def __init__(self):
        super(FontSiameseNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 512)
        self.fcOut = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x) # 256
        x = x.view(x.size()[0], -1)

        x = self.sigmoid(self.fc(x))
        return x

    def forward_pair(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)

        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x
