import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class AdaptiveColorBrightnessLoss(nn.Module):
    def __init__(self):
        super(AdaptiveColorBrightnessLoss, self).__init__()
        # Add learnable parameters for color balance
        self.color_weight = nn.Parameter(torch.tensor(0.08))  # Lower initial value
        self.saturation_limit = nn.Parameter(torch.tensor(1.2))  # Limit over-saturation
        
    def forward(self, output, target):
        # 色彩豐富度指標 (Color richness metric)
        output_rgb = output.permute(0, 2, 3, 1)
        target_rgb = target.permute(0, 2, 3, 1)
        
        # For output image
        rg_out = output_rgb[..., 0] - output_rgb[..., 1]
        yb_out = 0.5 * (output_rgb[..., 0] + output_rgb[..., 1]) - output_rgb[..., 2]
        std_rg_out = torch.std(rg_out, dim=[1, 2])
        std_yb_out = torch.std(yb_out, dim=[1, 2])
        colorfulness_out = torch.sqrt(std_rg_out**2 + std_yb_out**2)
        
        # For target image - reference for colorfulness
        rg_target = target_rgb[..., 0] - target_rgb[..., 1]
        yb_target = 0.5 * (target_rgb[..., 0] + target_rgb[..., 1]) - target_rgb[..., 2]
        std_rg_target = torch.std(rg_target, dim=[1, 2])
        std_yb_target = torch.std(yb_target, dim=[1, 2])
        colorfulness_target = torch.sqrt(std_rg_target**2 + std_yb_target**2)
        
        # Optimize to match target colorfulness but with a small boost
        target_color = torch.clamp(colorfulness_target * self.saturation_limit, max=1.5)
        color_diff = torch.abs(colorfulness_out - target_color)
        color_loss = color_diff.mean()
        
        # Brightness preservation - multiple levels
        # Overall brightness
        brightness_diff_global = torch.abs(output.mean() - target.mean())
        
        # Channel-wise brightness
        brightness_diff_channels = torch.abs(output.mean(dim=[2, 3]) - target.mean(dim=[2, 3])).mean()
        
        # Local brightness preservation (using pooled patches)
        output_patches = F.avg_pool2d(output, kernel_size=16, stride=8)
        target_patches = F.avg_pool2d(target, kernel_size=16, stride=8)
        brightness_diff_local = torch.abs(output_patches - target_patches).mean()
        
        # Combined brightness loss
        brightness_loss = 0.4 * brightness_diff_global + 0.4 * brightness_diff_channels + 0.2 * brightness_diff_local
        
        return self.color_weight * color_loss + 0.05 * brightness_loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36]
        self.loss_model = vgg.to(device).eval()
        self.feature_layers = [5, 12, 22, 32]  # Use multiple feature layers
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        
        # Extract features from multiple layers
        loss = 0.0
        features_true = []
        features_pred = []
        
        # Store activations at different layers
        activation_true = y_true
        activation_pred = y_pred
        
        for i, layer in enumerate(self.loss_model):
            activation_true = layer(activation_true)
            activation_pred = layer(activation_pred)
            
            if i in self.feature_layers:
                # Normalize features
                norm_true = F.normalize(activation_true.view(activation_true.size(0), -1))
                norm_pred = F.normalize(activation_pred.view(activation_pred.size(0), -1))
                
                # Calculate loss with layer-specific weights (deeper layers get higher weight)
                layer_weight = 0.5 + 0.5 * (i / max(self.feature_layers))
                layer_loss = F.mse_loss(norm_true, norm_pred) * layer_weight
                loss += layer_loss
                
        return loss / len(self.feature_layers)

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return 40.0 - torch.mean(psnr)

def detail_loss(y_true, y_pred):
    # Extract high-frequency details using Laplacian filter
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32).view(1, 1, 3, 3).to(y_true.device)
    
    # Apply to each channel separately
    details_true = torch.cat([
        F.conv2d(y_true[:, i:i+1], laplacian_kernel, padding=1) 
        for i in range(y_true.size(1))
    ], dim=1)
    
    details_pred = torch.cat([
        F.conv2d(y_pred[:, i:i+1], laplacian_kernel, padding=1) 
        for i in range(y_pred.size(1))
    ], dim=1)
    
    # Focus on preserving details (high-frequency components)
    return F.l1_loss(details_true, details_pred)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def adaptive_histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    # Calculate importance weights based on image characteristics
    edges = detect_edges(y_true)
    importance = torch.sigmoid(edges * 5.0) + 0.5  # Higher weights for edge regions
    
    # Weighted histogram calculation
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    
    # Gaussian kernel for smooth histograms
    def gaussian_kernel(x, mu, sigma):
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Calculate histogram with importance weighting
    y_true_flat = y_true.reshape(-1, 1)
    y_pred_flat = y_pred.reshape(-1, 1)
    importance_flat = importance.reshape(-1, 1)
    
    y_true_hist = torch.sum(gaussian_kernel(y_true_flat, bin_edges, sigma) * importance_flat, dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred_flat, bin_edges, sigma) * importance_flat, dim=0)
    
    # Normalize histograms
    y_true_hist /= y_true_hist.sum() + 1e-8
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    
    # Calculate Earth Mover's Distance (approximation)
    cdf_true = torch.cumsum(y_true_hist, dim=0)
    cdf_pred = torch.cumsum(y_pred_hist, dim=0)
    emd = torch.sum(torch.abs(cdf_true - cdf_pred))
    
    return emd

def detect_edges(image):
    # Simple Sobel edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    
    # Apply to grayscale version of the image
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    return torch.sqrt(grad_x**2 + grad_y**2)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.color_brightness_loss = AdaptiveColorBrightnessLoss()
        
        # Weights are learnable to adapt during training
        self.log_weight_l1 = nn.Parameter(torch.tensor(0.0))    # Smooth L1 (initialized as exp(0)=1.0)
        self.log_weight_perc = nn.Parameter(torch.tensor(-1.9)) # Perceptual (initialized as exp(-1.9)≈0.15)
        self.log_weight_hist = nn.Parameter(torch.tensor(-3.0)) # Histogram (initialized as exp(-3.0)≈0.05)
        self.log_weight_ssim = nn.Parameter(torch.tensor(-0.22)) # MS-SSIM (initialized as exp(-0.22)≈0.8)
        self.log_weight_psnr = nn.Parameter(torch.tensor(-5.3)) # PSNR (initialized as exp(-5.3)≈0.005)
        self.log_weight_color = nn.Parameter(torch.tensor(-2.3)) # Color (initialized as exp(-2.3)≈0.1)
        self.log_weight_detail = nn.Parameter(torch.tensor(-3.7)) # Detail (new, initialized as exp(-3.7)≈0.025)

    def forward(self, y_true, y_pred):
        # Convert log weights to actual weights (ensures positivity and better gradient behavior)
        weight_l1 = torch.exp(self.log_weight_l1)
        weight_perc = torch.exp(self.log_weight_perc)
        weight_hist = torch.exp(self.log_weight_hist)
        weight_ssim = torch.exp(self.log_weight_ssim)
        weight_psnr = torch.exp(self.log_weight_psnr)
        weight_color = torch.exp(self.log_weight_color)
        weight_detail = torch.exp(self.log_weight_detail)
        
        # Calculate individual loss components
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        hist_l = adaptive_histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_bright_l = self.color_brightness_loss(y_true, y_pred)
        detail_l = detail_loss(y_true, y_pred)

        # Combine losses with learned weights
        total_loss = (weight_l1 * smooth_l1_l + 
                      weight_perc * perc_l + 
                      weight_hist * hist_l + 
                      weight_ssim * ms_ssim_l + 
                      weight_psnr * psnr_l + 
                      weight_color * color_bright_l +
                      weight_detail * detail_l)
        
        return torch.mean(total_loss)