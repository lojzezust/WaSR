import torch
from torch import nn

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, last_arm=False):
        super(AttentionRefinementModule, self).__init__()

        self.last_arm = last_arm

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x

        x = self.global_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        weights = self.sigmoid(x)

        out = weights * input

        if self.last_arm:
            weights = self.global_pool(out)
            out = weights * out

        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, bg_channels, sm_channels, num_features):
        super(FeatureFusionModule, self).__init__()

        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(num_features, num_features, 1)
        self.conv3 = nn.Conv2d(num_features, num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_big, x_small):
        if x_big.size(2) > x_small.size(2):
            x_small = self.upsampling(x_small)

        x = torch.cat((x_big, x_small), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_out = self.relu(x)

        x = self.global_pool(conv1_out)
        x = self.conv2(x)
        x = self.conv3(x)
        weights = self.sigmoid(x)

        mul = weights * conv1_out
        out = conv1_out + mul

        return out

class ASPPv2Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, bias=False, bn=False, relu=False):
        modules = []
        modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=bias))

        if bn:
            modules.append(nn.BatchNorm2d(out_channels))

        if relu:
            modules.append(nn.ReLU())

        super(ASPPv2Conv, self).__init__(*modules)

class ASPPv2(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, relu=False, biased=True):
        super(ASPPv2, self).__init__()
        modules = []

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPv2Conv(in_channels, out_channels, rate, bias=True))

        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Sum convolution results
        res = torch.stack(res).sum(0)
        return res
