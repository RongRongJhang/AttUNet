import torch
import torch.nn as nn
import torch.nn.functional as F
from ecb import ECB

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class EfficientAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(EfficientAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attn = EfficientAttention(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.depthwise_conv.bias, 0)

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, padding=0)
        self.attn = EfficientAttention(channels)
        self.gelu = nn.GELU()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return self.gelu(x)

class Denoiser(nn.Module):
    def __init__(self, num_filters=40, kernel_size=3):  # 減少濾波器數量
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.attn = EfficientAttention(num_filters)  # 替換 MultiHeadSelfAttention
        self.fusion4 = FeatureFusion(num_filters)
        self.fusion3 = FeatureFusion(num_filters)
        self.fusion2 = FeatureFusion(num_filters)
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.refine4 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='gelu', with_idt=True)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='gelu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='gelu', with_idt=True)
        self.res_layer = nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=1)
        self.output_layer = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=1)
        self.gelu = nn.GELU()
        self._init_weights()

    def forward(self, x):
        x1 = self.gelu(self.conv1(x))
        x2 = self.gelu(self.conv2(x1))
        x3 = self.gelu(self.conv3(x2))
        x4 = self.gelu(self.conv4(x3))
        x = self.attn(x4)
        x = self.up4(x)
        x = self.fusion4(x, x3)
        x = self.refine4(x)
        x = self.up3(x)
        x = self.fusion3(x, x2)
        x = self.refine3(x)
        x = self.up2(x)
        x = self.fusion2(x, x1)
        x = self.refine2(x)
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.res_layer, self.output_layer]:
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='gelu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=40):  # 減少濾波器數量
        super(LYT, self).__init__()
        self.denoiser_rgb = Denoiser(filters, kernel_size=3)
        self.channel_adjust = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, act_type='gelu', with_idt=True)
        self.final_adjustments = nn.Sequential(
            nn.Conv2d(filters, filters // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(filters // 2, 3, kernel_size=3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):
        rgb_denoised = self.denoiser_rgb(inputs) + inputs
        adjusted = self.channel_adjust(rgb_denoised)
        ref = self.msef(adjusted)
        recombined = self.recombine(ref)
        refined = self.final_refine(F.gelu(recombined))
        output = self.final_adjustments(refined)
        return torch.sigmoid(output)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='gelu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)