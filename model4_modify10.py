import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, stride=stride, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=bias
        )
        self._init_weights()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

    def _init_weights(self):
        init.kaiming_uniform_(self.depthwise.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.pointwise.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.depthwise.bias is not None:
            init.constant_(self.depthwise.bias, 0)
        if self.pointwise.bias is not None:
            init.constant_(self.pointwise.bias, 0)

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
        # 編碼器
        self.conv1 = SeparableConv2d(2, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = SeparableConv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = SeparableConv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.conv4 = SeparableConv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv5 = SeparableConv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.conv6 = SeparableConv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        # 解碼器
        self.refine6 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine5 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine4 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine3 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine2 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.refine1 = SeparableConv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.output_layer = nn.Conv2d(num_filters, 2, kernel_size=kernel_size, padding=1)
        self.activation = getattr(F, activation)
        self._init_weights()

    def forward(self, x):
        input_size = x.shape[2:]  # 保存輸入尺寸
        # 編碼器
        x1 = self.activation(self.conv1(x))  # 256 x 256
        x2 = self.activation(self.conv2(x1))  # 128 x 128
        x3 = self.activation(self.conv3(x2))  # 128 x 128
        x4 = self.activation(self.conv4(x3))  # 64 x 64
        x5 = self.activation(self.conv5(x4))  # 64 x 64
        x6 = self.activation(self.conv6(x5))  # 32 x 32
        # 瓶頸
        x = self.bottleneck(x6)  # 32 x 32
        # 解碼器
        x = self.up6(x)  # 64 x 64
        x = self.refine6(self.activation(x + x5))  # 匹配 x5
        x = self.up5(x)  # 128 x 128
        x = self.refine5(self.activation(x + x3))  # 匹配 x3
        x = self.up4(x)  # 256 x 256
        x = self.refine4(self.activation(x + x1))  # 匹配 x1
        x = self.refine3(self.activation(x))
        x = self.refine2(self.activation(x))
        x = self.refine1(self.activation(x))  # [1, 16, 256, 256]
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)  # 確保尺寸
        return torch.tanh(self.output_layer(x))  # [1, 2, 256, 256]

    def _init_weights(self):
        init.kaiming_uniform_(self.output_layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.output_layer.bias is not None:
            init.constant_(self.output_layer.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.process_y = self._create_processing_layers(filters)
        self.process_cb = self._create_processing_layers(filters)
        self.process_cr = self._create_processing_layers(filters)
        self.denoiser_cbcr = Denoiser(filters // 2, kernel_size=3, activation='relu')
        self.lum_pool = nn.MaxPool2d(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=8)
        self.lum_up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.lum_conv = nn.Conv2d(filters, filters, kernel_size=1, padding=0)
        self.ref_conv = nn.Conv2d(filters * 2, filters, kernel_size=1, padding=0)
        self.msef1 = MSEFBlock(filters)
        self.msef2 = MSEFBlock(filters)
        self.msef3 = MSEFBlock(filters)
        self.recombine = SeparableConv2d(filters * 2, filters, kernel_size=3, padding=1)
        self.final_refine = SeparableConv2d(filters, filters, kernel_size=3, padding=1)
        self.final_adjustments = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self._init_weights()

    def _create_processing_layers(self, filters):
        return nn.Sequential(
            SeparableConv2d(1, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SeparableConv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _rgb_to_ycbcr(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        yuv = torch.stack((y, u, v), dim=1)
        return yuv
    
    def _rgb_to_oklab(self, image):
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        eps = 1e-6
        l_ = torch.sign(l) * (torch.abs(l) + eps).pow(1/3)
        m_ = torch.sign(m) * (torch.abs(m) + eps).pow(1/3)
        s_ = torch.sign(s) * (torch.abs(s) + eps).pow(1/3)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_out = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        oklab = torch.stack((L, a, b_out), dim=1)
        return oklab

    def forward(self, inputs):
        input_size = inputs.shape[2:]  # 保存輸入尺寸
        ycbcr = self._rgb_to_oklab(inputs)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cbcr = torch.cat([cb, cr], dim=1)
        cbcr_denoised = self.denoiser_cbcr(cbcr)  # 輸出 2 通道
        cb_denoised, cr_denoised = torch.split(cbcr_denoised, 1, dim=1)
        y_processed = self.process_y(y)
        cb_processed = self.process_cb(cb_denoised)
        cr_processed = self.process_cr(cr_denoised)
        ref = torch.cat([cb_processed, cr_processed], dim=1)
        lum = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum = lum + lum_1
        ref = self.ref_conv(ref)
        shortcut = ref
        ref = ref + 0.2 * self.lum_conv(lum)
        ref = self.msef1(ref)
        ref = self.msef2(ref)
        ref = self.msef3(ref)
        ref = ref + shortcut
        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        refined = self.final_refine(F.relu(recombined))
        output = self.final_adjustments(refined)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        return torch.sigmoid(output)
    
    def _init_weights(self):
        for module in self.children():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)