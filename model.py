import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from ecb import ECB

def rgb_to_hsv(tensor):
    """
    將 RGB 圖像轉換為 HSV 空間
    輸入: tensor [batch_size, 3, height, width], 範圍 [0, 1]
    輸出: tensor [batch_size, 3, height, width], HSV 格式
    """
    # 確保輸入範圍在 [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    r, g, b = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]

    # 計算 V (Value)
    v, max_idx = torch.max(tensor, dim=1)
    v_min = torch.min(tensor, dim=1)[0]
    s = torch.where(v > 0, (v - v_min) / (v + 1e-8), torch.zeros_like(v))

    # 計算 H (Hue)
    h = torch.zeros_like(v)
    mask_r = max_idx == 0
    mask_g = max_idx == 1
    mask_b = max_idx == 2

    # H 計算公式
    h = torch.where(mask_r, (g - b) / (v - v_min + 1e-8), h)
    h = torch.where(mask_g, 2.0 + (b - r) / (v - v_min + 1e-8), h)
    h = torch.where(mask_b, 4.0 + (r - g) / (v - v_min + 1e-8), h)
    h = h * 60.0 / 360.0  # 將角度轉換為 [0, 1]
    h = torch.where(h < 0, h + 1.0, h)  # 處理負值
    h = torch.where(v_min == v, torch.zeros_like(h), h)  # 若 v_min == v，H 設為 0

    # 堆疊 H, S, V 通道
    return torch.stack([h, s, v], dim=1)

def hsv_to_rgb(tensor):
    """
    將 HSV 圖像轉換為 RGB 空間
    輸入: tensor [batch_size, 3, height, width], HSV 格式
    輸出: tensor [batch_size, 3, height, width], 範圍 [0, 1]
    """
    # 確保輸入範圍在 [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    h, s, v = tensor[:, 0, :, :], tensor[:, 1, :, :], tensor[:, 2, :, :]

    # 將 H 從 [0, 1] 轉換為角度 [0, 6]
    h = h * 6.0
    i = torch.floor(h).long()
    f = h - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # 初始化 RGB 通道
    r = torch.zeros_like(v)
    g = torch.zeros_like(v)
    b = torch.zeros_like(v)

    # 根據 i 的值選擇 RGB 計算公式
    mask_i0 = (i % 6) == 0
    mask_i1 = (i % 6) == 1
    mask_i2 = (i % 6) == 2
    mask_i3 = (i % 6) == 3
    mask_i4 = (i % 6) == 4
    mask_i5 = (i % 6) == 5

    r = torch.where(mask_i0, v, r)
    r = torch.where(mask_i1, q, r)
    r = torch.where(mask_i2, p, r)
    r = torch.where(mask_i3, p, r)
    r = torch.where(mask_i4, t, r)
    r = torch.where(mask_i5, v, r)

    g = torch.where(mask_i0, t, g)
    g = torch.where(mask_i1, v, g)
    g = torch.where(mask_i2, v, g)
    g = torch.where(mask_i3, q, g)
    g = torch.where(mask_i4, p, g)
    g = torch.where(mask_i5, p, g)

    b = torch.where(mask_i0, p, b)
    b = torch.where(mask_i1, p, b)
    b = torch.where(mask_i2, t, b)
    b = torch.where(mask_i3, v, b)
    b = torch.where(mask_i4, v, b)
    b = torch.where(mask_i5, q, b)

    # 堆疊 RGB 通道
    return torch.stack([r, g, b], dim=1)

# 通道注意力模組
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# 色彩增強模組
class ColorEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super(ColorEnhanceBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attn = ChannelAttention(channels)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.channel_attn(x)
        return x + residual

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

# 修改後的 MSEFBlock
class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.eca_attn = ECABlock(filters)
        self.channel_attn = ChannelAttention(filters)  # 新增通道注意力
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        x1 = self.depthwise_conv(x_norm)
        x2 = self.eca_attn(x_norm)
        x3 = self.channel_attn(x_norm)  # 應用通道注意力
        x_fused = x1 * x2 * x3
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

# 修改後的 Denoiser
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
        self.color_enhance = ColorEnhanceBlock(num_filters)  # 新增色彩增強模組
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.output_layer = nn.Conv2d(3, 3, kernel_size=kernel_size, padding=1)
        self.res_layer = nn.Conv2d(num_filters, 3, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

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
        x = self.color_enhance(x)  # 應用色彩增強模組
        x = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))

    def activation_wrapper(self, x):
        if self.activation == F.relu:
            return self.activation(x, inplace=True)
        return self.activation(x)

    def _init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

# 修改後的 LYT
class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.denoiser_rgb = Denoiser(filters, kernel_size=3, activation='relu')
        self.channel_adjust = nn.Conv2d(3, filters, kernel_size=3, padding=1)
        self.msef = MSEFBlock(filters)
        self.recombine = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0, act_type='relu', with_idt=True)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    def forward(self, inputs):
        # 將輸入轉換為 HSV 空間
        inputs_hsv = rgb_to_hsv(inputs)  # 使用自定義函數
        # 去噪處理（在 HSV 空間）
        hsv_denoised = self.denoiser_rgb(inputs_hsv) + inputs_hsv
        # 轉回 RGB 空間進行後續處理
        rgb_denoised = hsv_to_rgb(hsv_denoised)  # 使用自定義函數
        # 調整通道數
        adjusted = self.channel_adjust(rgb_denoised)
        # MSEFBlock 處理
        ref = self.msef(adjusted)
        # 後續處理
        recombined = self.recombine(ref)
        refined = self.final_refine(F.relu(recombined))
        output = self.final_adjustments(refined)
        return torch.sigmoid(output)

    def _init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)