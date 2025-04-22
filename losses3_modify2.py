import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:36]  # 使用更深層特徵
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

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

def color_constancy_loss(y_true, y_pred, lambda_var=0.1):
    """
    顏色恆常性損失：計算RGB通道均值和方差的MSE。
    Args:
        y_true: 真實圖像，形狀 [batch_size, 3, height, width]
        y_pred: 預測圖像，形狀 [batch_size, 3, height, width]
        lambda_var: 控制方差損失的權重
    Returns:
        顏色恆常性損失值
    """
    # 計算每個通道的均值
    mean_true = torch.mean(y_true, dim=[2, 3])  # [batch_size, 3]
    mean_pred = torch.mean(y_pred, dim=[2, 3])  # [batch_size, 3]
    mean_loss = F.mse_loss(mean_pred, mean_true)

    # 計算每個通道的標準差（方差的平方根）
    var_true = torch.std(y_true, dim=[2, 3])  # [batch_size, 3]
    var_pred = torch.std(y_pred, dim=[2, 3])  # [batch_size, 3]
    var_loss = F.mse_loss(var_pred, var_true)

    # 總顏色恆常性損失
    return mean_loss + lambda_var * var_loss

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(device)
        self.alpha1 = 1.00  # Smooth L1
        self.alpha2 = 0.20  # 感知損失
        self.alpha3 = 0.05  # 直方圖損失
        self.alpha4 = 0.80  # MS-SSIM
        self.alpha5 = 0.005 # PSNR
        self.alpha6 = 0.10  # 顏色恆常性損失（新加入）

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_constancy_loss(y_true, y_pred, lambda_var=0.1)

        total_loss = (self.alpha1 * smooth_l1_l + 
                      self.alpha2 * perc_l + 
                      self.alpha3 * hist_l + 
                      self.alpha4 * ms_ssim_l + 
                      self.alpha5 * psnr_l + 
                      self.alpha6 * color_l)
        return torch.mean(total_loss)