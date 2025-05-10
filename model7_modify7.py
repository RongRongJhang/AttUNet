import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ecb import ECB

# model7_modify5 精簡化

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

class SimplifiedAttention(nn.Module):
    def __init__(self, channels):
        super(SimplifiedAttention, self).__init__()
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

class LightweightMSEFBlock(nn.Module):
    def __init__(self, filters):
        super(LightweightMSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.attention = SimplifiedAttention(filters)  # 使用簡化注意力
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

class EfficientDenoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3):
        super(EfficientDenoiser, self).__init__()
        # 使用深度可分離卷積減少參數
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_filters, 1),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1, groups=num_filters)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 1),
            nn.Conv2d(num_filters, num_filters, kernel_size, stride=2, padding=1, groups=num_filters)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 1),
            nn.Conv2d(num_filters, num_filters, kernel_size, stride=2, padding=1, groups=num_filters)
        )
        self.bottleneck = LightweightMSEFBlock(num_filters)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=0.5, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=0.5, act_type='relu', with_idt=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._init_weights()

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = F.relu(self.conv3(x2), inplace=True)
        
        x = self.bottleneck(x3)
        
        x = self.up3(x)
        x = self.refine3(x + x2)
        x = self.up2(x)
        x = self.refine2(x + x1)
        
        return x
    
    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=32):  # 減少通道數
        super(LYT, self).__init__()
        self.denoiser = EfficientDenoiser(filters, kernel_size=3)
        self.msef = LightweightMSEFBlock(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=0.5, 
                               act_type='relu', with_idt=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters//2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters//2, 3, 3, padding=1)
        )
        self._init_weights()

    def forward(self, inputs):
        # Enhanced denoising
        denoised = self.denoiser(inputs)
        
        # Multi-scale feature enhancement
        enhanced = self.msef(denoised)
        
        # Final refinement
        refined = self.final_refine(enhanced) + enhanced
        
        # Generate output
        output = self.final_conv(refined)
        
        # Global residual with learned adjustment
        output = output + inputs
        
        # Use tanh for wider output range (-1 to 1) then scale to (0,1)
        return (torch.tanh(output) + 1) / 2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)