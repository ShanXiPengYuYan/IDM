import torch
from torch import nn
from einops import rearrange
from .SaCaCrossMamba import SaCaCrossMamba


class S2CrossMamba(nn.Module):
    def __init__(self, num_classes=6, AuHu=None, HS_c=None, Lidar_c=None):
        super(S2CrossMamba, self).__init__()

        self.AuHu = AuHu
        self.conv3d_hy1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )

        if AuHu:
            self.conv2d_li1 = nn.Sequential(
                nn.Conv2d(in_channels=Lidar_c, out_channels=64, kernel_size=(3, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        else:
            self.conv2d_li1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

        self.conv2d_h1 = nn.Sequential(
            nn.Conv2d(in_channels=(HS_c-2)*4, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # nn.Dropout(p=0.3),
        self.FC = nn.Sequential(nn.Linear(64, num_classes))

        # self.beta = nn.Parameter(torch.tensor(0.5),requires_grad=True)
        self.w1 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.w3 = nn.Parameter(torch.rand(1), requires_grad=True)

        self.GP = nn.AdaptiveAvgPool2d(1)

        self.SaCaCrossMamba = SaCaCrossMamba(hidden_dim=64)
        # self.Mode_nossm=Mode_nossm(hidden_dim=64)

    # 384
    def forward(self, x1, x2, label, flag, **kwargs):
        x1 = x1.unsqueeze(1)
        # # Lidar支路
        if self.AuHu:
            x_l_1 = self.conv2d_li1(x2)  # 64 32 9 9
        else:
            x_l_1 = self.conv2d_li1(x2)  # 64 32 9 9

        x_h_1 = self.conv3d_hy1(x1)  # 64 16 28 9 9
        x_h_2 = rearrange(x_h_1, 'b c h w y ->b (c h) w y')  # 64 224 9 9
        x_h_3 = self.conv2d_h1(x_h_2)
        x_out, out_dict = self.SaCaCrossMamba(x_h_3, x_l_1)

        # x_out = self.Mode_nossm(x_h_3, x_l_1)
        x_out = self.GP(x_out.permute(0, 3, 1, 2))

        x_out = x_out.squeeze(2).squeeze(2)  # 64 384


        # x1 output
        x_out1, out_dict1 = self.SaCaCrossMamba(x_h_3, torch.zeros_like(x_l_1))
        x_out2, out_dict2 = self.SaCaCrossMamba(torch.zeros_like(x_h_3), x_l_1)

        x_out1 = self.GP(x_out1.permute(0, 3, 1, 2))
        x_out1 = x_out1.squeeze(2).squeeze(2)  # 64 384

        x_out2 = self.GP(x_out2.permute(0, 3, 1, 2))
        x_out2 = x_out2.squeeze(2).squeeze(2)  # 64 384

        x_out = self.FC(x_out)

        x_out1 = self.FC(x_out1)
        x_out2 = self.FC(x_out2)


        out_dict = {
            'feature': x_out.reshape(x_out.size(0), -1),
            'output1': x_out1,
            'output2': x_out2,
        }


        return x_out, out_dict
