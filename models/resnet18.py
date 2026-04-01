import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .backbone import resnet18

class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        # self.fc_out = nn.Linear(input_dim, output_dim)
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, x, y):
        feature = torch.cat((x, y), dim=1)
        # feature = x+y
        output = self.fc_out(feature)
        return x, y, feature, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=128, dim=128, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class resnet(nn.Module):
    def __init__(self, n_classes, band_MultiModal):
        super(resnet, self).__init__()
        self.n_classes = n_classes
        self.modal1 = resnet18(modality='hs', channel=band_MultiModal[0])
        self.modal2 = resnet18(modality='lidar', channel=band_MultiModal[1])

        self.fusion_module = ConcatFusion(128 + 128, output_dim=n_classes)


    def forward(self, hs, lidar, labels=None, flag='train', **kwargs):
        if flag == 'train':
            self.train()
        else:
            self.eval()

        # 编码器提取特征
        hs_spatial = self.modal1(hs)  # [B, 128, H, W]
        lidar_spatial = self.modal2(lidar)  # [B, 128, H, W]

        # 全局平均池化，得到样本级特征向量
        hs_pooled = F.adaptive_avg_pool2d(hs_spatial, 1)
        lidar_pooled = F.adaptive_avg_pool2d(lidar_spatial, 1)

        hs_feature = torch.flatten(hs_pooled, 1)  # [B, 128]
        lidar_feature = torch.flatten(lidar_pooled, 1)  # [B, 128]
        # 融合并分类
        _, _, feature, output = self.fusion_module(hs_feature, lidar_feature)

        out_dict = {
            'output': output,
            'feature': feature,
            'feature1': hs_feature,
            'feature2': lidar_feature,
            'hs_spatial': hs_spatial,  # [B, 128, H, W] - 用于梯度调制
            'lidar_spatial': lidar_spatial,  # [B, 128, H, W]
        }

        return output, out_dict



