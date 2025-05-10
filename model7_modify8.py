import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ecb import ECB

class GroupNormalization(nn.Module):
    def __init__(self, dim):
        super(GroupNormalization, self).__init__()
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=dim, affine=True)

    def forward(self, x):
        return self.norm(x) # x shape: (N, C, H, W), c = dim

class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.group_norm = GroupNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn = SEBlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.group_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class ColorBrightnessAdjustment(nn.Module):
    def __init__(self, channels):
        super(ColorBrightnessAdjustment, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//4, 3, padding=1)
        self.conv2 = nn.Conv2d(channels//4, 4, 3, padding=1)  # Output RGB + brightness
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        adj = self.conv2(x)
        color_weights = self.sigmoid(adj[:, :3]) * 1.5  # Scale to enhance color (0-1.5 range)
        brightness_map = self.sigmoid(adj[:, 3:]) + 0.5  # Center around 1.0 (0.5-1.5 range)
        return color_weights, brightness_map
    
    def _init_weights(self):
        init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MSEFBlock(num_filters)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x), inplace=True)
        x2 = self.activation(self.conv2(x1), inplace=True)
        x3 = self.activation(self.conv3(x2), inplace=True)
        x = self.bottleneck(x3)
        x = self.up3(x)
        x = self.refine3(x + x2)
        x = self.up2(x)
        x = self.refine2(x + x1)
        
        return x
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LaaFNet(nn.Module):
    def __init__(self, filters=32):
        super(LaaFNet, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.msef = MSEFBlock(filters)
        self.color_brightness_adjust = ColorBrightnessAdjustment(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters//2, 3, 3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):      
        denoised = self.denoiser(inputs)
        enhanced = self.msef(denoised)
        color_weights, brightness_map = self.color_brightness_adjust(enhanced)
        refined = self.final_refine(enhanced) + enhanced
        output = self.final_conv(refined)
        output = output * brightness_map  # Brightness adjustment
        output = output * (1 + color_weights)  # Color enhancement
        output = output + inputs

        return (torch.tanh(output) + 1) / 2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)