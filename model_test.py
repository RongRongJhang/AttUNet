import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from ecb import ECB

class LayerNormalization(nn.Module):
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x * self.gamma + self.beta

class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                              padding=(kernel_size - 1) // 2, bias=False, groups=channels)
        self._init_weights()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels, 1)
        y = self.conv(y)
        y = torch.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y

    def _init_weights(self):
        init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')

class ColorAttention(nn.Module):
    def __init__(self, channels):
        super(ColorAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x) * 2.0

class BrightnessEnhancer(nn.Module):
    def __init__(self, channels):
        super(BrightnessEnhancer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//8, 3, padding=1),  # 減少通道數
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels//8, 3, padding=1),  # 增加多尺度感知
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 1, 1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        brightness_map = self.conv(x) * self.gamma
        return brightness_map + 0.5

class EfficientAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(EfficientAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.key_conv = nn.Conv2d(channels, channels // reduction, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query_conv(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch, -1, height * width)
        value = self.value_conv(x).view(batch, -1, height * width)
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        return out + x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.pointwise.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise.bias, 0)
        init.constant_(self.pointwise.bias, 0)

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = DepthwiseSeparableConv(filters, filters)
        self.eca_attn = ECABlock(filters)
        self.color_attn = ColorAttention(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.eca_attn(x_norm)
        x_fused = x1 * x2
        color_weights = self.color_attn(x_fused)
        x_out = x_fused + x
        return x_out, color_weights
    
    def _init_weights(self):
        init.constant_(self.depthwise_conv.depthwise.bias, 0)
        init.constant_(self.depthwise_conv.pointwise.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = DepthwiseSeparableConv(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = DepthwiseSeparableConv(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = DepthwiseSeparableConv(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = EfficientAttention(num_filters)
        self.refine4 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=0.5, act_type='relu', with_idt=True)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=0.5, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=0.5, act_type='relu', with_idt=True)
        self.upconv4 = nn.PixelShuffle(upscale_factor=2)
        self.upconv3 = nn.PixelShuffle(upscale_factor=2)
        self.upconv2 = nn.PixelShuffle(upscale_factor=2)
        self.channel_adjust4 = nn.Conv2d(num_filters // 4, num_filters, 1)
        self.channel_adjust3 = nn.Conv2d(num_filters // 4, num_filters, 1)
        self.channel_adjust2 = nn.Conv2d(num_filters // 4, num_filters, 1)
        self.brightness_enhancer = BrightnessEnhancer(num_filters)
        self.brightness_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x), inplace=True)
        x2 = self.activation(self.conv2(x1), inplace=True)
        x3 = self.activation(self.conv3(x2), inplace=True)
        x4 = self.activation(self.conv4(x3), inplace=True)
        
        x = self.bottleneck(x4)
        brightness_map = self.brightness_enhancer(x)
        brightness_map = self.brightness_upsample(brightness_map)
        
        x = self.upconv4(x)
        x = self.channel_adjust4(x)
        x = self.refine4(x + x3)
        x = self.upconv3(x)
        x = self.channel_adjust3(x)
        x = self.refine3(x + x2)
        x = self.upconv2(x)
        x = self.channel_adjust2(x)
        x = self.refine2(x + x1)
        
        return x, brightness_map
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            init.kaiming_uniform_(layer.depthwise.weight, a=0, mode='fan_in', nonlinearity='relu')
            init.kaiming_uniform_(layer.pointwise.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.depthwise.bias is not None:
                init.constant_(layer.depthwise.bias, 0)
            if layer.pointwise.bias is not None:
                init.constant_(layer.pointwise.bias, 0)
        for layer in [self.channel_adjust4, self.channel_adjust3, self.channel_adjust2]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=64):  # 增加 filters 到 64
        super(LYT, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.msef = MSEFBlock(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=0.5, 
                               act_type='relu', with_idt=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, 3, 3, padding=1)
        )
        self.color_adjust = nn.Sequential(  # 增強 color_adjust
            nn.Conv2d(3, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)
        )
        self._init_weights()

    def forward(self, inputs):
        input_size = inputs.size()[2:]
        denoised, brightness_map = self.denoiser(inputs)
        enhanced, color_weights = self.msef(denoised)
        refined = self.final_refine(enhanced) + enhanced
        output = self.final_conv(refined)
        brightness_map = F.interpolate(brightness_map, size=input_size, mode='bilinear', align_corners=True)
        output = output * brightness_map
        color_weights = F.interpolate(color_weights, size=input_size, mode='bilinear', align_corners=True)
        output = output * (1 + color_weights)
        output = self.color_adjust(output + inputs)
        return (torch.tanh(output) + 1) / 2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)