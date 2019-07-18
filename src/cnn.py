import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=params['conv1_c'],
                               kernel_size=params['conv1_k'], stride=params['conv1_s'],
                               padding=params['conv1_p'])

        self.pool1 = nn.MaxPool2d(kernel_size=params['pool1_k'], stride=params['pool1_s'],
                                  padding=params['pool1_p'])

        self.conv2 = nn.Conv2d(in_channels=params['conv1_c'], out_channels=params['conv2_c'],
                               kernel_size=params['conv2_k'], stride=params['conv2_s'],
                               padding=params['conv2_p'])

        self.pool2 = nn.MaxPool2d(kernel_size=params['pool2_k'], stride=params['pool2_s'],
                                  padding=params['pool2_p'])

        self.conv3 = nn.Conv2d(in_channels=params['conv2_c'], out_channels=params['conv3_c'],
                               kernel_size=params['conv3_k'], stride=params['conv3_s'],
                               padding=params['conv3_p'])

        self.conv3_bn = nn.BatchNorm2d(params['conv3_c'])

        self.conv4 = nn.Conv2d(in_channels=params['conv3_c'], out_channels=params['conv4_c'],
                               kernel_size=params['conv4_k'], stride=params['conv4_s'],
                               padding=params['conv4_p'])

        self.pool3 = nn.MaxPool2d(kernel_size=params['pool3_k'], stride=params['pool3_s'],
                                  padding=params['pool3_p'])

        self.conv5 = nn.Conv2d(in_channels=params['conv4_c'], out_channels=params['conv5_c'],
                               kernel_size=params['conv5_k'], stride=params['conv5_s'],
                               padding=params['conv5_p'])

        self.conv5_bn = nn.BatchNorm2d(params['conv5_c'])

        self.pool4 = nn.MaxPool2d(kernel_size=params['pool4_k'], stride=params['pool4_s'],
                                  padding=params['pool4_p'])

        self.conv6 = nn.Conv2d(in_channels=params['conv5_c'], out_channels=params['conv6_c'],
                               kernel_size=params['conv6_k'], stride=params['conv6_s'],
                               padding=params['conv6_p'])

        self.conv6_bn = nn.BatchNorm2d(params['conv6_c'])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = self.pool4(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        return x
