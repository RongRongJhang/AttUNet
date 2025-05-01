import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class EnhancedColorBrightnessLoss(nn.Module):
    def __init__(self):
        super(EnhancedColorBrightnessLoss, self).__init__()
        
    def forward(self, output, target):
        output_rgb = output.permute(0, 2, 3, 1)
        rg = output_rgb[..., 0] - output_rgb[..., 1]
        yb = 0.5 * (output_rgb[..., 0] + output_rgb[..., 1]) - output_rgb[..., 2]
        std_rg = torch.std(rg, dim=[1, 2])
        std_yb = torch.std(yb, dim=[1, 2])
        colorfulness = torch.sqrt(std_rg**2 + std_yb**2)
        color_loss = -colorfulness.mean()
        brightness_diff = torch.abs(output.mean() - target.mean())
        return 0.1 * color_loss + 0.05 * brightness_diff

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36]
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, y_true, y_pred, device):
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        edge_true_x = F.conv2d(y_true.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        edge_true_y = F.conv2d(y_true.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        edge_pred_x = F.conv2d(y_pred.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        edge_pred_y = F.conv2d(y_pred.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        edge_true = torch.sqrt(edge_true_x**2 + edge_true_y**2 + 1e-8)
        edge_pred = torch.sqrt(edge_pred_x**2 + edge_pred_y**2 + 1e-8)
        return F.mse_loss(edge_true, edge_pred)

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=128, sigma=0.01):  # 減少 bins
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_true_hist /= y_true_hist.sum() + 1e-8
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.color_brightness_loss = EnhancedColorBrightnessLoss()
        self.edge_loss = EdgeLoss()
        
        # 調整後的權重參數
        self.alpha1 = 1.00  # Smooth L1
        self.alpha2 = 0.20  # 感知損失 (增加)
        self.alpha3 = 0.05  # 直方圖損失
        self.alpha4 = 0.80  # MS-SSIM
        self.alpha5 = 0.01  # PSNR (增加)
        self.alpha6 = 0.08  # 色彩/亮度損失 (降低)
        self.alpha7 = 0.10  # 新增邊緣損失

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_bright_l = self.color_brightness_loss(y_true, y_pred)
        edge_l = self.edge_loss(y_true, y_pred, y_true.device)

        total_loss = (self.alpha1 * smooth_l1_l + 
                      self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + 
                      self.alpha4 * ms_ssim_l + 
                      self.alpha5 * psnr_l + 
                      self.alpha6 * color_bright_l +
                      self.alpha7 * edge_l)
        
        return torch.mean(total_loss)