import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ecb import ECB

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x * self.gamma + self.beta

class ShuffleAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ShuffleAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.group_channels = channels // reduction
        
        self.groups = max(1, channels // self.group_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 修改權重形狀以匹配分組後的通道數
        self.weight = nn.Parameter(torch.zeros(1, self.group_channels, 1, 1))
        self.bias = nn.Parameter(torch.ones(1, self.group_channels, 1, 1))
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        y = self.avg_pool(x)  # [b, c, 1, 1]
        y = y.reshape(b * self.groups, self.group_channels, 1, 1)
        y = self.sigmoid(y * self.weight + self.bias)
        
        # Spatial attention
        x_group = x.reshape(b * self.groups, self.group_channels, h, w)
        out = x_group * y
        out = out.reshape(b, c, h, w)
        
        return out

    def _init_weights(self):
        init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')

class ColorBrightnessAdjustment(nn.Module):
    def __init__(self, channels):
        super(ColorBrightnessAdjustment, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels//4, 3, padding=1)
        self.conv2 = nn.Conv2d(channels//4, 4, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        adj = self.conv2(x)
        color_weights = self.sigmoid(adj[:, :3]) * 1.5
        brightness_map = self.sigmoid(adj[:, 3:]) + 0.5
        return color_weights, brightness_map
    
    def _init_weights(self):
        init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attention = ShuffleAttention(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.attention(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

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
    def __init__(self, filters=48):
        super(LaaFNet, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.msef = MSEFBlock(filters)
        self.color_brightness_adjust = ColorBrightnessAdjustment(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, 
                               act_type='relu', with_idt=True)
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