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

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

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

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.eca_attn = ECABlock(filters)
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.eca_attn(x_norm)
        x_fused = x1 * x2
        x_out = x_fused + x
        return x_out
    
    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.depthwise_conv.bias, 0)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0
        self.head_dim = embed_size // num_heads
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self._init_weights()

    def split_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        x = x.reshape(batch_size, height * width, -1)
        query = self.split_heads(self.query_dense(x), batch_size)
        key = self.split_heads(self.key_dense(x), batch_size)
        value = self.split_heads(self.value_dense(x), batch_size)
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(batch_size, -1, self.embed_size)
        output = self.combine_heads(attention)
        return output.reshape(batch_size, height, width, self.embed_size).permute(0, 3, 1, 2)

    def _init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias, 0)
        init.constant_(self.key_dense.bias, 0)
        init.constant_(self.value_dense.bias, 0)
        init.constant_(self.combine_heads.bias, 0)

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super(Denoiser, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.refine4 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output_layer = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def activation_wrapper(self, x):
        if self.activation == F.relu:
            return self.activation(x, inplace=True)
        return self.activation(x)

    def forward(self, x):
        x1 = self.activation_wrapper(self.conv1(x))
        x2 = self.activation_wrapper(self.conv2(x1))
        x3 = self.activation_wrapper(self.conv3(x2))
        x4 = self.activation_wrapper(self.conv4(x3))
        x = self.bottleneck(x4)
        x = self.up4(x)
        x = self.refine4(x + x3)
        x = self.up3(x)
        x = self.refine3(x + x2)
        x = self.up2(x)
        x = self.refine2(x + x1)
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

# New Color Enhancement Module
class ColorEnhancementModule(nn.Module):
    def __init__(self, channels):
        super(ColorEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.channel_attention = ECABlock(channels)
        self.gamma = nn.Parameter(torch.ones(1))
        self._init_weights()
        
    def forward(self, x):
        # Enhance contrast first
        x_enhanced = F.relu(self.conv1(x))
        x_enhanced = self.conv2(x_enhanced)
        # Apply channel attention for color enhancement
        x_enhanced = self.channel_attention(x_enhanced)
        # Weighted residual connection for controllable enhancement
        return x + self.gamma * x_enhanced
    
    def _init_weights(self):
        init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        init.constant_(self.conv2.bias, 0)

# Enhanced brightness adjustment
class BrightnessEnhancementModule(nn.Module):
    def __init__(self, channels):
        super(BrightnessEnhancementModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.brightness_scale = nn.Parameter(torch.tensor([1.2]))  # Adjustable brightness parameter
        self._init_weights()
        
    def forward(self, x):
        attention_map = self.spatial_attention(x)
        # Apply brightness enhancement with spatial adaptivity
        x_brightened = self.conv(x) * self.brightness_scale
        return x + attention_map * x_brightened
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.denoiser_rgb = Denoiser(filters, kernel_size=3, activation='relu')
        self.channel_adjust = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.msef = MSEFBlock(filters)
        
        # New modules for enhanced color and brightness
        self.color_enhancement = ColorEnhancementModule(filters)
        self.brightness_enhancement = BrightnessEnhancementModule(filters)
        
        self.recombine = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        
        # Separate channel-wise processing for RGB
        self.r_adjust = nn.Conv2d(filters, 1, kernel_size=1)
        self.g_adjust = nn.Conv2d(filters, 1, kernel_size=1)
        self.b_adjust = nn.Conv2d(filters, 1, kernel_size=1)
        
        # Global color contrast enhancement
        self.color_contrast = nn.Parameter(torch.tensor([1.2]))  # Control global color contrast
        self.color_bias = nn.Parameter(torch.zeros(3, 1, 1))     # Color channel bias
        
        self._init_weights()

    def forward(self, inputs):
        # Denoising
        rgb_denoised = self.denoiser_rgb(inputs) + inputs
        
        # Channel adjustment
        adjusted = self.channel_adjust(rgb_denoised)
        
        # MSEF processing
        ref = self.msef(adjusted)
        
        # Color enhancement
        color_enhanced = self.color_enhancement(ref)
        
        # Brightness enhancement
        brightness_enhanced = self.brightness_enhancement(color_enhanced)
        
        # Recombine and refine
        recombined = self.recombine(brightness_enhanced)
        refined = self.final_refine(F.relu(recombined)) + recombined
        
        # Separate RGB channel processing
        r = self.r_adjust(refined)
        g = self.g_adjust(refined)
        b = self.b_adjust(refined)
        
        # Combine RGB channels
        rgb_output = torch.cat([r, g, b], dim=1)
        
        # Apply color contrast enhancement and bias
        enhanced_output = rgb_output * self.color_contrast + self.color_bias
        
        # Add residual connection from input
        final_output = enhanced_output + inputs
        
        # Use scaled tanh for wider dynamic range instead of sigmoid
        return torch.tanh(final_output) * 1.2
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)