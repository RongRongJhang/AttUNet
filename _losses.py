import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        # 使用淺層特徵，捕捉紋理和邊緣
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:12]  # 改為 features[:12]
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true = y_true.to(next(self.loss_model.parameters()).device)
        y_pred = y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))

def edge_loss(y_true, y_pred):
    # Sobel 算子計算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=y_true.device).view(1, 1, 3, 3)
    
    # 對每個通道計算梯度
    grad_true_x = torch.zeros_like(y_true)
    grad_true_y = torch.zeros_like(y_true)
    grad_pred_x = torch.zeros_like(y_pred)
    grad_pred_y = torch.zeros_like(y_pred)
    
    for c in range(y_true.size(1)):
        grad_true_x[:, c:c+1] = F.conv2d(y_true[:, c:c+1], sobel_x, padding=1)
        grad_true_y[:, c:c+1] = F.conv2d(y_true[:, c:c+1], sobel_y, padding=1)
        grad_pred_x[:, c:c+1] = F.conv2d(y_pred[:, c:c+1], sobel_x, padding=1)
        grad_pred_y[:, c:c+1] = F.conv2d(y_pred[:, c:c+1], sobel_y, padding=1)
    
    # 計算 x 和 y 方向的梯度 MSE
    loss_x = F.mse_loss(grad_true_x, grad_pred_x)
    loss_y = F.mse_loss(grad_true_y, grad_pred_y)
    return (loss_x + loss_y) / 2.0

def color_loss(y_true, y_pred):
    oklab_true = rgb_to_oklab(y_true)
    oklab_pred = rgb_to_oklab(y_pred)
    # 分離亮度（L）和色度（a, b）
    l_true = oklab_true[:, 0:1, :, :]  # L 通道
    l_pred = oklab_pred[:, 0:1, :, :]
    ab_true = oklab_true[:, 1:, :, :]  # a 和 b 通道
    ab_pred = oklab_pred[:, 1:, :, :]
    # 亮度損失權重 0.6，色度損失權重 0.4
    l_loss = torch.mean(torch.abs(l_true - l_pred))
    ab_loss = torch.mean(torch.abs(ab_true - ab_pred))
    return 0.6 * l_loss + 0.4 * ab_loss

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

def histogram_loss(y_true, y_pred, bins=128, sigma=0.02):  # 減少 bins，增加 sigma
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)
    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=[0, 1, 2, 3])
    y_true_hist /= y_true_hist.sum() + 1e-8
    y_pred_hist /= y_pred_hist.sum() + 1e-8
    return torch.mean(torch.abs(y_true_hist - y_pred_hist))

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
    def __init__(self, device, total_epochs=1000):
        super(CombinedLoss, self).__init__()
        # 使用多層 VGG 特徵（conv1_2, conv2_2, conv3_2）
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.perceptual_loss_model = nn.ModuleList([
            vgg[:4],   # conv1_2
            vgg[:9],   # conv2_2
            vgg[:18],  # conv3_2
        ]).to(device).eval()
        for module in self.perceptual_loss_model:
            for param in module.parameters():
                param.requires_grad = False
        
        # 調整權重
        self.alpha1 = 1.0     # Smooth L1
        self.alpha2 = 0.2     # 感知損失（略提高）
        self.alpha3 = 0.05    # 直方圖損失
        self.alpha4 = 0.8     # MS-SSIM
        self.alpha5 = 0.05    # PSNR（提高）
        self.alpha6 = 0.4     # 色彩損失
        self.alpha7 = 0.2     # 邊緣損失（新增）

        # 訓練總輪次，默認值為 100
        self.total_epochs = total_epochs
        self.current_epoch = 0  # 當前輪次，初始為 0

    def adjust_loss_weights(self, epoch):
        """動態調整損失函數的權重"""
        if epoch < 0.2 * self.total_epochs:
            self.alpha1, self.alpha4 = 1.2, 1.0
            self.alpha2, self.alpha7 = 0.1, 0.1
            self.alpha3, self.alpha5, self.alpha6 = 0.05, 0.05, 0.5
        elif epoch < 0.6 * self.total_epochs:
            self.alpha1, self.alpha2, self.alpha3 = 1.0, 0.2, 0.05
            self.alpha4, self.alpha5, self.alpha6 = 0.8, 0.05, 0.6
            self.alpha7 = 0.2
        else:
            self.alpha1, self.alpha3, self.alpha4 = 1.0, 0.05, 0.8
            self.alpha2, self.alpha5, self.alpha6 = 0.3, 0.05, 0.5
            self.alpha7 = 0.3

    def set_epoch(self, epoch):
        """設置當前輪次並更新權重"""
        self.current_epoch = epoch
        self.adjust_loss_weights(epoch)

    def forward(self, y_true, y_pred):
        # 像素級損失
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        
        # 感知損失（多層特徵）
        perc_l = 0.0
        for layer in self.perceptual_loss_model:
            perc_l += F.mse_loss(layer(y_true), layer(y_pred))
        perc_l /= len(self.perceptual_loss_model)  # 平均多層損失
        
        # 直方圖損失
        hist_l = histogram_loss(y_true, y_pred)
        
        # PSNR 損失
        psnr_l = psnr_loss(y_true, y_pred)
        
        # 色彩損失
        color_l = color_loss(y_true, y_pred)
        
        # 邊緣損失
        edge_l = edge_loss(y_true, y_pred)

        # 總損失
        total_loss = (self.alpha1 * smooth_l1_l +
                      self.alpha2 * perc_l +
                      self.alpha3 * hist_l +
                      self.alpha4 * ms_ssim_l +
                      self.alpha5 * psnr_l +
                      self.alpha6 * color_l +
                      self.alpha7 * edge_l)
        return torch.mean(total_loss)