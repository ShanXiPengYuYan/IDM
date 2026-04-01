# 导入所需的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset


# 定义模型类
class CO_CNN(nn.Module):
    def __init__(self, num_patches, n_class):
        super(CO_CNN, self).__init__()
        # 定义HSI通道的卷积层

        self.channel1 = num_patches[0]
        self.channel2 = num_patches[1]

        self.x1_conv_w1 = nn.Conv2d(in_channels=self.channel1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.x1_bn1 = nn.BatchNorm2d(16)

        self.x1_conv_w2 = nn.Conv2d(16, 32, kernel_size=1)
        self.x1_bn2 = nn.BatchNorm2d(32)
        self.x1_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.x1_conv_w3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.x1_bn3 = nn.BatchNorm2d(64)

        self.x1_conv_w31 = nn.Conv2d(64, 128, kernel_size=1)
        self.x1_bn31 = nn.BatchNorm2d(128)
        self.x1_pool31 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.x1_conv_w4 = nn.Conv2d(128, 128, kernel_size=1)
        self.x1_bn4 = nn.BatchNorm2d(128)

        self.x1_conv_w5 = nn.Conv2d(128, 64, kernel_size=1)
        self.x1_bn5 = nn.BatchNorm2d(64)
        self.x1_pool5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 定义LiDAR通道的卷积层
        self.x2_conv_w1 = nn.Conv2d(in_channels=self.channel2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.x2_bn1 = nn.BatchNorm2d(16)

        self.x2_conv_w2 = nn.Conv2d(16, 32, kernel_size=1)
        self.x2_bn2 = nn.BatchNorm2d(32)
        self.x2_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.x2_conv_w3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.x2_bn3 = nn.BatchNorm2d(64)

        self.x2_conv_w31 = nn.Conv2d(64, 128, kernel_size=1)
        self.x2_bn31 = nn.BatchNorm2d(128)
        self.x2_pool31 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.x2_conv_w4 = nn.Conv2d(128, 128, kernel_size=1)
        self.x2_bn4 = nn.BatchNorm2d(128)

        self.x2_conv_w5 = nn.Conv2d(128, 64, kernel_size=1)
        self.x2_bn5 = nn.BatchNorm2d(64)
        self.x2_pool5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # 联合编码层
        self.joint_conv = nn.Conv2d(128, n_class, kernel_size=1)
        # self.fc_out = nn.Linear(128, n_class)

    def forward(self, x1, x2, label, flag, **kwargs):
        if flag == 'train':
            self.train()
        else:
            self.eval()
        # x1通道（HSI）
        x1 = x1.view(-1, self.channel1, 13, 13)
        x1 = F.relu(self.x1_bn1(self.x1_conv_w1(x1)))
        x1 = F.relu(self.x1_bn2(self.x1_conv_w2(x1)))
        x1 = self.x1_pool2(x1)
        x1 = F.relu(self.x1_bn3(self.x1_conv_w3(x1)))
        x1 = F.relu(self.x1_bn31(self.x1_conv_w31(x1)))
        x1 = self.x1_pool31(x1)
        x1 = F.relu(self.x1_bn4(self.x1_conv_w4(x1)))
        x1 = F.relu(self.x1_bn5(self.x1_conv_w5(x1)))
        x1 = self.x1_pool5(x1)

        # x2通道（LiDAR）
        x2 = x2.view(-1, self.channel2, 13, 13)
        x2 = F.relu(self.x2_bn1(self.x2_conv_w1(x2)))
        x2 = F.relu(self.x2_bn2(self.x2_conv_w2(x2)))
        x2 = self.x2_pool2(x2)
        x2 = F.relu(self.x2_bn3(self.x2_conv_w3(x2)))
        x2 = F.relu(self.x2_bn31(self.x2_conv_w31(x2)))
        x2 = self.x2_pool31(x2)
        x2 = F.relu(self.x2_bn4(self.x2_conv_w4(x2)))
        x2 = F.relu(self.x2_bn5(self.x2_conv_w5(x2)))
        x2 = self.x2_pool5(x2)

        # 联合编码
        joint_layer = torch.cat([x1, x2], dim=1)
        x = self.joint_conv(joint_layer)
        output1 = self.joint_conv(torch.cat([x1, torch.zeros_like(x2)], dim=1)).view(x.size(0), -1)
        output2 = self.joint_conv(torch.cat([torch.zeros_like(x1), x2], dim=1)).view(x.size(0), -1)
        x = x.view(x.size(0), -1)

        outdict = {
            'feature': joint_layer.reshape(joint_layer.size(0), -1),
            'feature1': x1.reshape(x1.size(0), -1),
            'feature2': x2.reshape(x2.size(0), -1),
            'output1': output1,
            'output2': output2,
        }
        return x, outdict
