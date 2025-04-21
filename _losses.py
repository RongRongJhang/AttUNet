import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]  # 使用VGG16淺層特徵
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def sobel_edge_loss(y_true, y_pred):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    
    y_true_gray = y_true.mean(dim=1, keepdim=True)
    y_pred_gray = y_pred.mean(dim=1, keepdim=True)
    
    edge_x_true = F.conv2d(y_true_gray, sobel_x, padding=1)
    edge_y_true = F.conv2d(y_true_gray, sobel_y, padding=1)
    edge_x_pred = F.conv2d(y_pred_gray, sobel_x, padding=1)
    edge_y_pred = F.conv2d(y_pred_gray, sobel_y, padding=1)
    
    return F.mse_loss(edge_x_true, edge_x_pred) + F.mse_loss(edge_y_true, edge_y_pred)

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(torch.exp(-0.5 * ((y_true.unsqueeze(-1) - bin_edges) / sigma) ** 2), dim=[0, 1, 2, 3])
    y_pred_hist = torch.sum(torch.exp(-0.5 * ((y_pred.unsqueeze(-1) - bin_edges) / sigma) ** 2), dim=[0, 1, 2, 3])
    y_true_hist /= y_true_hist.sum() + 1e-8
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.alpha1 = 1.00  # Smooth L1
        self.alpha2 = 0.30  # 感知損失
        self.alpha3 = 0.02  # 直方圖損失
        self.alpha4 = 0.90  # MS-SSIM
        self.alpha5 = 0.01  # PSNR
        self.alpha6 = 0.10  # 邊緣損失

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        edge_l = sobel_edge_loss(y_true, y_pred)

        total_loss = (self.alpha1 * smooth_l1_l +
                      self.alpha2 * perc_l +
                      self.alpha3 * hist_l +
                      self.alpha4 * ms_ssim_l +
                      self.alpha5 * psnr_l +
                      self.alpha6 * edge_l)
        return torch.mean(total_loss)