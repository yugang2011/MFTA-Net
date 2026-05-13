import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.nf = 784
        self.conv_co = nn.Conv2d(in_channels=self.nf*2, out_channels=self.nf, kernel_size=1, stride=1)
        self.ca = ChannelAttention(in_planes=self.nf)
        self.sa = SpatialAttention()
        self.conv_fuse = nn.Conv2d(in_channels=self.nf*2, out_channels=self.nf, kernel_size=1, stride=1)
        self.conv_out = nn.Conv2d(in_channels=self.nf, out_channels=self.nf, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        co = self.conv_co(torch.cat([x, y], dim=1))
        ca_co = self.ca(co)
        y_ = y*ca_co
        sa_co = self.sa(co)
        y_ = y_*sa_co
        x_ = self.conv_fuse(torch.cat([x, y_], dim=1)) + x
        x_ = self.conv_out(x_)
        return x_

