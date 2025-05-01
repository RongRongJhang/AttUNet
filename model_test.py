import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from ecb import ECB

class LayerNormalization(nn.Module):
    # Keep original LayerNormalization as it seems standard channel-wise LN
    def __init__(self, dim):
        super(LayerNormalization, self).__init__()
        self.norm = nn.LayerNorm(dim)
        # Using learnable affine parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1) # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2) # B, C, H, W
        return x * self.gamma + self.beta

class ECABlock(nn.Module):
    # Keep original ECABlock
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # Use Conv1d with groups=channels for efficiency (depthwise-like)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        # Note: Original code had Conv1d(channels, channels, groups=channels)
        # A simpler implementation uses Conv1d(1, 1, ...) applied after reshape
        self.channels = channels # Store channels for reshape
        self._init_weights()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, 1, channels) # B, 1, C
        # Apply 1D convolution across channels
        y = self.conv(y) # B, 1, C
        y = torch.sigmoid(y).view(batch_size, channels, 1, 1) # B, C, 1, 1
        return x * y.expand_as(x)

    def _init_weights(self):
        # Kaiming initialization for conv weight
        init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))


class ColorAttention(nn.Module):
    def __init__(self, channels):
        super(ColorAttention, self).__init__()
        # Simpler attention mechanism, could experiment with reduction factor
        reduction = 8
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 3, 1, bias=False), # Output 3 channels for RGB
            nn.Sigmoid() # Output weights between 0 and 1
        )
        # Initialize last layer's weights small, bias towards 1 (if bias=True)
        # init.zeros_(self.conv[-2].weight) # Start with small weights
        init.kaiming_normal_(self.conv[0].weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv[2].weight, mode='fan_out', nonlinearity='sigmoid')


    def forward(self, x):
        # Output weights meant to scale RGB channels, centered around 1
        # Sigmoid * 2 maps to [0, 2], Sigmoid * 1.5 + 0.25 maps to [0.25, 1.75] etc.
        # Let's try scaling towards 1 initially.
        return self.conv(x) * 2.0 # Scale weights to be around 1 (range [0, 2])

# BrightnessEnhancer Removed

class MSEFBlock(nn.Module):
    def __init__(self, filters):
        super(MSEFBlock, self).__init__()
        self.layer_norm = LayerNormalization(filters)
        # Use depthwise separable convolution for efficiency
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters, bias=False)
        self.pointwise_conv = nn.Conv2d(filters, filters, kernel_size=1, bias=False) # Added pointwise
        self.eca_attn = ECABlock(filters)
        self.color_attn = ColorAttention(filters)
        self.act = nn.LeakyReLU(0.2, inplace=True) # Use LeakyReLU consistently
        self._init_weights()

    def forward(self, x):
        x_norm = self.layer_norm(x)
        # Path 1: Depthwise Separable Conv
        x1 = self.pointwise_conv(self.depthwise_conv(x_norm))
        # Path 2: ECA Attention
        x2 = self.eca_attn(x_norm)

        # Fusion using Addition (more robust)
        x_fused = self.act(x1 + x2) # Apply activation after fusion

        # Calculate color weights based on fused features
        color_weights = self.color_attn(x_fused) # Shape (B, 3, H, W)

        # Residual connection
        x_out = x_fused + x
        return x_out, color_weights # Return features and color weights

    def _init_weights(self):
        # Initialize weights (example)
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0.2, nonlinearity='relu')
        init.kaiming_uniform_(self.pointwise_conv.weight, a=0.2, nonlinearity='relu')
        # Biases are False here

# MultiHeadSelfAttention Removed from Denoiser Bottleneck

class Denoiser(nn.Module):
    def __init__(self, num_filters, kernel_size=3, activation='relu', bottleneck_blocks=4):
        super(Denoiser, self).__init__()

        self.activation = getattr(F, activation)

        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)

        # Bottleneck: Replace MHSA with ECB blocks
        self.bottleneck = nn.Sequential(
            *[ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type=activation, with_idt=True)
              for _ in range(bottleneck_blocks)]
        )

        # Refinement blocks (keep ECB)
        self.refine4 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type=activation, with_idt=True)
        self.refine3 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type=activation, with_idt=True)
        self.refine2 = ECB(inp_planes=num_filters, out_planes=num_filters, depth_multiplier=1.0, act_type=activation, with_idt=True)

        # Upsampling layers
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # BrightnessEnhancer and brightness_upsample removed

        self._init_weights()

    def forward(self, x):
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))

        # Bottleneck processing
        x_bottle = self.bottleneck(x4)

        # Decoder path with skip connections
        d4 = self.up4(x_bottle)
        # Ensure channels match for addition (should be num_filters)
        d4 = self.refine4(d4 + x3)

        d3 = self.up3(d4)
        d3 = self.refine3(d3 + x2)

        d2 = self.up2(d3)
        d2 = self.refine2(d2 + x1) # Output of decoder

        # No brightness map returned here
        return d2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class LYT(nn.Module):
    def __init__(self, filters=48):
        super(LYT, self).__init__()
        self.denoiser = Denoiser(filters, kernel_size=3, activation='relu', bottleneck_blocks=4)
        self.msef = MSEFBlock(filters)
        self.final_refine = ECB(inp_planes=filters, out_planes=filters, depth_multiplier=1.0,
                                act_type='relu', with_idt=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, 3, 3, padding=1) # Output 3 channels before final adjustment
        )
        # Keep the final color adjustment layer
        self.color_adjust = nn.Conv2d(3, 3, 1)
        # Initialize final_conv and color_adjust towards identity or small values if needed
        self._init_weights()


    def forward(self, inputs):
        input_size = inputs.size()[2:] # H, W

        # Denoising features from U-Net like structure
        denoised_features = self.denoiser(inputs) # (B, filters, H, W)

        # Multi-scale feature enhancement + Color Attention weights
        enhanced_features, color_weights = self.msef(denoised_features) # (B, filters, H, W), (B, 3, H, W)

        # Apply color weights to features *before* final refinement/conv
        # Ensure color_weights are broadcastable to enhanced_features if needed
        # Current color_weights (B, 3, H, W) cannot directly multiply (B, filters, H, W)
        # Option 1: Apply color_weights later to the 3-channel output
        # Option 2: Modify ColorAttention to output 'filters' channels (less intuitive)
        # Let's choose Option 1: Apply later

        # Final refinement
        refined = self.final_refine(enhanced_features) + enhanced_features # Residual connection

        # Map to 3 channels
        output_base = self.final_conv(refined) # (B, 3, H, W)

        # Apply learned color attention weights
        # Interpolate color_weights if their size differs from output_base (shouldn't here)
        # color_weights = F.interpolate(color_weights, size=input_size, mode='bilinear', align_corners=True)
        output_colored = output_base * color_weights # (B, 3, H, W) * (B, 3, H, W)

        # Global residual connection (original input + processed)
        # Apply the final linear color adjustment
        output = self.color_adjust(output_colored + inputs)

        # Output activation: Map to [0, 1]
        return (torch.tanh(output) + 1) / 2

    def _init_weights(self):
        # Standard initialization (as in original code)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Consider He initialization (kaiming_normal_) for LeakyReLU
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight) # For linear layers if any
                if m.bias is not None:
                   init.constant_(m.bias, 0)
        # Special init for color_adjust? Maybe initialize towards identity transformation
        # init.constant_(self.color_adjust.weight, 0) # Start with adding zero adjustment
        # init.constant_(self.color_adjust.bias, 0)
        # Or initialize weights to approximate identity for the R,G,B channels respectively
        # TBD based on performance