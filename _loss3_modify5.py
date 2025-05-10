import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class EnhancedColorLoss(nn.Module):
    """
    Calculates the difference in colorfulness between output and target.
    Ref: Hasler and Suesstrunk, "Measuring colorfulness in natural images" (2003)
    """
    def __init__(self, eps=1e-6):
        super(EnhancedColorLoss, self).__init__()
        self.eps = eps

    def calculate_colorfulness(self, img):
        # img shape: (B, C, H, W), range [0, 1]
        img_rgb = img.permute(0, 2, 3, 1) # B, H, W, C
        rg = img_rgb[..., 0] - img_rgb[..., 1]
        yb = 0.5 * (img_rgb[..., 0] + img_rgb[..., 1]) - img_rgb[..., 2]

        std_rg = torch.std(rg, dim=[1, 2])
        std_yb = torch.std(yb, dim=[1, 2])
        mean_rg = torch.mean(rg, dim=[1, 2])
        mean_yb = torch.mean(yb, dim=[1, 2])

        # Combine standard deviations and means
        colorfulness = torch.sqrt(std_rg**2 + std_yb**2 + self.eps) + 0.3 * torch.sqrt(mean_rg**2 + mean_yb**2 + self.eps)
        return colorfulness # Shape: (B,)

    def forward(self, output, target):
        output_colorfulness = self.calculate_colorfulness(output)
        target_colorfulness = self.calculate_colorfulness(target)

        # Use L1 loss to minimize the difference
        color_loss = F.l1_loss(output_colorfulness, target_colorfulness)
        return color_loss

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        # Use VGG19 up to conv4_4 (layer 35 in features, index 34)
        # weights=models.VGG19_Weights.IMAGENET1K_V1 is an alternative if DEFAULT causes issues
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:35]
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False
        # Normalization parameters for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def normalize(self, tensor):
        # Input tensor is assumed to be in [0, 1] range
        return (tensor - self.mean) / self.std

    def forward(self, y_pred, y_true): # Note: Swapped order to match common LPIPS/Perceptual loss usage
        y_pred_norm = self.normalize(y_pred)
        y_true_norm = self.normalize(y_true)
        
        pred_features = self.loss_model(y_pred_norm)
        true_features = self.loss_model(y_true_norm)

        # Using L1 loss on features often works well for perceptual loss
        return F.l1_loss(pred_features, true_features)

def psnr_loss(y_true, y_pred, max_val=1.0):
    """Calculates PSNR loss. Aim to maximize PSNR -> minimize this loss."""
    mse = F.mse_loss(y_true, y_pred, reduction='mean')
    # Prevent log10(0)
    if mse == 0:
        # PSNR is infinite, loss should be minimal (or negative infinity if not clamped)
        # Return a large negative number or a indicator that it's perfect.
        # Using a very small loss value if perfect.
         return torch.tensor(0.0, device=mse.device, dtype=mse.dtype) # Or a small negative number like -40.0
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + 1e-9)) # Add epsilon for stability
    # We want to maximize PSNR. A common way is to return -PSNR or (Target_PSNR - PSNR).
    # Using 40 as a target reference, similar to original code. Higher PSNR means lower loss.
    return 40.0 - psnr # Removed torch.mean() as mse_loss reduction='mean' already averages.

def smooth_l1_loss(y_true, y_pred, beta=1.0):
    return F.smooth_l1_loss(y_pred, y_true, beta=beta, reduction='mean') # Swapped order consistent with convention

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, **kwargs):
    # Ensure win_size is odd and <= min(H, W)
    # Default win_size in pytorch_msssim is 11
    # Add K1, K2 defaults similar to TF implementation
    return 1.0 - ms_ssim(y_pred, y_true, data_range=max_val, size_average=True, K=(0.01, 0.03), **kwargs) # Swapped order

def gaussian_kernel(x, mu, sigma):
    # Ensure dimensions are broadcastable
    # x: (B, C, H, W, 1)
    # mu: (bins,) -> (1, 1, 1, 1, bins)
    # sigma: scalar
    mu = mu.view(1, 1, 1, 1, -1)
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    device = y_true.device
    bin_edges = torch.linspace(0.0, 1.0, bins, device=device)

    # Process channel by channel or flatten? Flattening is simpler here.
    y_true_flat = y_true.reshape(y_true.size(0), -1, 1) # B, N, 1
    y_pred_flat = y_pred.reshape(y_pred.size(0), -1, 1) # B, N, 1

    # Calculate weighted histograms per image in batch
    true_hist_unnorm = torch.sum(gaussian_kernel(y_true_flat.unsqueeze(-1), bin_edges, sigma), dim=1) # B, bins
    pred_hist_unnorm = torch.sum(gaussian_kernel(y_pred_flat.unsqueeze(-1), bin_edges, sigma), dim=1) # B, bins

    # Normalize histograms per image
    true_hist = true_hist_unnorm / (true_hist_unnorm.sum(dim=1, keepdim=True) + 1e-8)
    pred_hist = pred_hist_unnorm / (pred_hist_unnorm.sum(dim=1, keepdim=True) + 1e-8)

    # Calculate L1 loss between histograms, average over batch
    return torch.mean(torch.sum(torch.abs(true_hist - pred_hist), dim=1))


class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.color_loss = EnhancedColorLoss()

        # Adjusted weights prioritizing PSNR/SSIM, slightly boosting Perceptual (for LPIPS)
        self.alpha1 = 1.00  # Smooth L1 (Pixel consistency) - Keep high
        self.alpha2 = 0.20  # Perceptual loss (LPIPS related) - Increased slightly
        self.alpha3 = 0.05  # Histogram loss (Distribution) - Keep low or moderate
        self.alpha4 = 0.90  # MS-SSIM (Structure) - Increased
        self.alpha5 = 0.08  # PSNR loss (Pixel fidelity) - Increased significantly
        self.alpha6 = 0.02  # Color difference loss - Reduced significantly / Optional

    def forward(self, y_pred, y_true): # Changed order to y_pred, y_true for consistency
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_pred, y_true)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = self.color_loss(y_pred, y_true)

        total_loss = (self.alpha1 * smooth_l1_l +
                      self.alpha2 * perc_l +
                      self.alpha3 * hist_l +
                      self.alpha4 * ms_ssim_l +
                      self.alpha5 * psnr_l +
                      self.alpha6 * color_l)

        # No need for torch.mean(total_loss) if individual losses are already averaged per batch element
        # Check if individual losses return per-element loss or mean loss
        # Assuming they return mean loss across batch (check implementations)
        return total_loss