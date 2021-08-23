import torch
from torch import nn
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from .layers import AttentionRefinementModule, FeatureFusionModule, ASPPv2

class NoIMUDecoder(nn.Module):
    """Decoder without IMU input."""
    def __init__(self, num_classes=3):
        super(NoIMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048)
        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512, last_arm=True),
            nn.Conv2d(512, 2048, 1) # Equalize number of features with ARM1
        )

        self.ffm = FeatureFusionModule(256, 2048, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], num_classes)

    def forward(self, x):
        features = x

        arm1 = self.arm1(features['out'])
        arm2 = self.arm2(features['skip2'])
        arm_combined = arm1 + arm2

        x = self.ffm(features['skip1'], arm_combined)

        output = self.aspp(x)

        return output

class IMUDecoder(nn.Module):
    """Decoder with IMU information merging."""
    def __init__(self, num_classes=3):
        super(IMUDecoder, self).__init__()

        self.arm1 = AttentionRefinementModule(2048 + 1)
        self.aspp1 = ASPPv2(2048, [6, 12, 18], 32)
        self.ffm1 = FeatureFusionModule(2048 + 1, 32, 1024)

        self.arm2 = nn.Sequential(
            AttentionRefinementModule(512 + 1, last_arm=True),
            nn.Conv2d(512 + 1, 1024, 1, bias=False) # Equalize number of features with FFM1
        )

        self.ffm = FeatureFusionModule(256 + 1, 1024, 1024)
        self.aspp = ASPPv2(1024, [6, 12, 18, 24], num_classes)

    def forward(self, x):
        features = x

        # Resize IMU mask to two required scales
        out = features['out']
        skip1 = features['skip1']
        imu_mask = features['imu_mask'].float().unsqueeze(1)
        imu_mask_s1 = TF.resize(imu_mask, (out.size(2), out.size(3)), InterpolationMode.NEAREST)
        imu_mask_s0 = TF.resize(imu_mask, (skip1.size(2), skip1.size(3)), InterpolationMode.NEAREST)

        # Concat backbone output and IMU
        out_imu = torch.cat([out, imu_mask_s1], dim=1)
        arm1 = self.arm1(out_imu)

        aspp1 = self.aspp1(out)

        # Fuse ARM1 and ASPP1
        ffm1 = self.ffm1(arm1, aspp1)

        # Concat Skip 2 and IMU
        skip2_imu = torch.cat([features['skip2'], imu_mask_s1], dim=1)
        arm2 = self.arm2(skip2_imu)

        arm_combined = ffm1 + arm2

        # Concat Skip 1 and IMU
        skip1_imu = torch.cat([features['skip1'], imu_mask_s0], dim=1)
        x = self.ffm(skip1_imu, arm_combined)

        output = self.aspp(x)

        return output
