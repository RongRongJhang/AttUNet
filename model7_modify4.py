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
            nn.Conv2d(channels//8, 3, 1),  # Output RGB attention
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.conv(x) * 2.0  # Scale to enhance color

class BrightnessEnhancer(nn.Module):
    def __init__(self, channels):
        super(BrightnessEnhancer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        brightness_map = self.conv(x) * self.gamma
        return brightness_map + 0.5  # Center around 1.0

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
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
        self.brightness_enhancer = BrightnessEnhancer(num_filters)
        self.brightness_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # New upsampling layer
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x), inplace=True)
        x2 = self.activation(self.conv2(x1), inplace=True)
        x3 = self.activation(self.conv3(x2), inplace=True)
        x4 = self.activation(self.conv4(x3), inplace=True)
        
        x = self.bottleneck(x4)
        brightness_map = self.brightness_enhancer(x)
        brightness_map = self.brightness_upsample(brightness_map)  # Upsample to match output size
        
        x = self.up4(x)
        x = self.refine4(x + x3)
        x = self.up3(x)
        x = self.refine3(x + x2)
        x = self.up2(x)
        x = self.refine2(x + x1)
        
        return x, brightness_map
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu')
        self.msef = MSEFBlock(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, 
                               act_type='relu', with_idt=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, 3, 3, padding=1)
        )
        self.color_adjust = nn.Conv2d(3, 3, 1)
        self._init_weights()

    def forward(self, inputs):
        # Get input size for proper resizing
        input_size = inputs.size()[2:]
        
        # Enhanced denoising with brightness awareness
        denoised, brightness_map = self.denoiser(inputs)
        
        # Multi-scale feature enhancement with color attention
        enhanced, color_weights = self.msef(denoised)
        
        # Final refinement
        refined = self.final_refine(enhanced) + enhanced
        
        # Generate output
        output = self.final_conv(refined)
        
        # Ensure brightness map matches output size
        brightness_map = F.interpolate(brightness_map, size=input_size, mode='bilinear', align_corners=True)
        
        # Apply brightness and color adjustments
        output = output * brightness_map  # Brightness adjustment
        color_weights = F.interpolate(color_weights, size=input_size, mode='bilinear', align_corners=True)
        output = output * (1 + color_weights)  # Color enhancement
        
        # Global residual with learned color adjustment
        output = self.color_adjust(output + inputs)
        
        # Use tanh for wider output range (-1 to 1) then scale to (0,1)
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