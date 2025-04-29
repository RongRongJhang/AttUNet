import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from ecb import ECB

# model7_modify5

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=0.25):
        super(InceptionBlock, self).__init__()
        # 計算各分支的通道數
        reduced_channels = int(in_channels * reduction)
        branch_channels = out_channels // 4

        # 分支 1: 1x1 卷積
        self.branch1 = nn.Conv2d(in_channels, branch_channels, kernel_size=1)

        # 分支 2: 1x1 卷積 + 3x3 卷積
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, branch_channels, kernel_size=3, padding=1)
        )

        # 分支 3: 1x1 卷積 + 5x5 卷積
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, branch_channels, kernel_size=5, padding=2)
        )

        # 分支 4: 3x3 最大池化 + 1x1 卷積
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1)
        )

        self._init_weights()

    def forward(self, x):
        # 並行計算各分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 在通道維度上拼接
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return F.relu(output)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)

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
                              padding=(kernel_size - 1) // 2, bias=False)
        self.channel_interaction = nn.Conv2d(channels, channels, kernel_size=1)  # 新增通道交互
        self._init_weights()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels, 1)
        y = self.conv(y)
        y = torch.sigmoid(y).view(batch_size, channels, 1, 1)
        x = self.channel_interaction(x)  # 增強通道間關係
        return x * y

    def _init_weights(self):
        init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.channel_interaction.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.constant_(self.channel_interaction.bias, 0)

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
        self.inception = InceptionBlock(num_filters, num_filters)  # 替換 conv2 為 Inception 模塊
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.refine4 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
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
        x1 = self.activation_wrapper(self.conv1(x))  # [batch_size, num_filters, H, W]
        x2 = self.inception(x1)  # [batch_size, num_filters, H, W]
        x3 = self.activation_wrapper(self.conv3(x2))  # [batch_size, num_filters, H//2, W//2]
        x4 = self.activation_wrapper(self.conv4(x3))  # [batch_size, num_filters, H//4, W//4]
        x = self.bottleneck(x4)  # [batch_size, num_filters, H//4, W//4]
        x = self.up4(x)  # [batch_size, num_filters, H//2, W//2]
        x = self.refine4(x + x3)  # [batch_size, num_filters, H//2, W//2]
        x = self.up3(x)  # [batch_size, num_filters, H, W]
        x = self.refine3(x + x2)  # [batch_size, num_filters, H, W]
        # 移除 self.up2(x)，因為 x 已經與 x1 尺寸匹配
        x = self.refine2(x + x1)  # [batch_size, num_filters, H, W]
        x = self.res_layer(x)  # [batch_size, 3, H, W]
        return torch.tanh(self.output_layer(x + x))  # [batch_size, 3, H, W]
    
    def _init_weights(self):
        for layer in [self.conv1, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.denoiser_rgb = Denoiser(filters, kernel_size=3, activation='relu')
        self.channel_adjust = nn.Conv2d(3, filters, kernel_size=3, padding=1)  # 新增通道調整層
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1))  # 在 __init__ 中定義
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))  # 在 __init__ 中定義
        self._init_weights()
    
    def forward(self, inputs):
        # 直接使用 RGB 輸入進行去噪
        rgb_denoised = self.denoiser_rgb(inputs) + inputs  # 去噪後的 RGB 圖片，形狀 [batch_size, 3, height, width]
        # 調整通道數
        adjusted = self.channel_adjust(rgb_denoised)  # 形狀 [batch_size, filters, height, width]
        # MSEFBlock 處理
        ref = self.msef(adjusted)
        # 後續處理
        recombined = self.recombine(ref)
        refined = self.final_refine(F.relu(recombined)) + recombined
        output = self.final_adjustments(refined) + inputs
        # 使用 __init__ 中定義的 scale 和 bias
        output = output * self.scale + self.bias
        return torch.clamp(output, 0, 1)  # 確保輸出在 [0, 1]
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)