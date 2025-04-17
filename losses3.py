import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:16]  # 更新 weights API
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def color_loss(y_true, y_pred):
    # 改進：考慮 Oklab 空間中的 ab 通道一致性
    oklab_true = rgb_to_oklab(y_true)  # 需要實現此函數
    oklab_pred = rgb_to_oklab(y_pred)
    ab_true = oklab_true[:, 1:, :, :]  # a 和 b 通道
    ab_pred = oklab_pred[:, 1:, :, :]
    return torch.mean(torch.abs(ab_true - ab_pred))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))  # 加入 eps 避免除零
    return 40.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_true_hist /= y_true_hist.sum() + 1e-8  # 避免除零
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

# 實現 rgb_to_oklab 以供損失函數使用
def rgb_to_oklab(image):
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
    return torch.stack((L, a, b_out), dim=1)

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36]  # 更深層特徵
        self.perceptual_loss_model = vgg.to(device).eval()
        for param in self.perceptual_loss_model.parameters():
            param.requires_grad = False
        self.alpha1 = 1.00    # Smooth L1
        self.alpha2 = 0.15    # 提升感知損失權重
        self.alpha3 = 0.05    # 直方圖損失
        self.alpha4 = 0.8     # 提升 MS-SSIM 權重
        self.alpha5 = 0.005   # 降低 PSNR 權重
        self.alpha6 = 0.4     # 提升色彩損失權重

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = F.mse_loss(self.perceptual_loss_model(y_true), self.perceptual_loss_model(y_pred))
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        oklab_true = rgb_to_oklab(y_true)
        oklab_pred = rgb_to_oklab(y_pred)
        color_l = torch.mean(torch.abs(oklab_true - oklab_pred))  # 考慮 L 通道

        total_loss = (self.alpha1 * smooth_l1_l + self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + self.alpha5 * psnr_l + 
                      self.alpha6 * color_l + self.alpha4 * ms_ssim_l)
        return torch.mean(total_loss)