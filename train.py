import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.utils import save_image
from model import LYT
from losses import CombinedLoss
from dataloader import create_dataloaders
import os
import numpy as np
from datetime import datetime
from measure import metrics
from tqdm import tqdm

def validate(model, dataloader, device, result_dir):
    model.eval()
    with torch.no_grad():
        for low, high, name in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            filename = name[0] if not name[0].endswith('.png') else name[0]
            save_path = os.path.join(result_dir, filename)
            save_image(output, save_path)

def main():
    # Hyperparameters
    # train_low = 'data/LOLv1/Train/input'
    # train_high = 'data/LOLv1/Train/target'
    # test_low = 'data/LOLv1/Test/input'
    # test_high = 'data/LOLv1/Test/target/'

    # train_low = 'data/LOLv2/Real_captured/Train/Low'
    # train_high = 'data/LOLv2/Real_captured/Train/Normal'
    # test_low = 'data/LOLv2/Real_captured/Test/Low'
    # test_high = 'data/LOLv2/Real_captured/Test/Normal'

    train_low = 'data/LOLv2/Synthetic/Train/Low'
    train_high = 'data/LOLv2/Synthetic/Train/Normal'
    test_low = 'data/LOLv2/Synthetic/Test/Low'
    test_high = 'data/LOLv2/Synthetic/Test/Normal'

    learning_rate = 2e-4 
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    result_dir = '/content/drive/MyDrive/Att-UNet/results/output/'

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = LYT().to(device)

    # criterion = CombinedLoss(device, total_epochs=num_epochs).to(device)
    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_psnr = 0
    best_ssim = 0
    best_lpips = 1
    
    print('Training started.')
    for epoch in range(num_epochs):

        # criterion.set_epoch(epoch)  # 在每個 epoch 開始時更新權重
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        validate(model, test_loader, device, result_dir)
        avg_psnr, avg_ssim, avg_lpips = metrics(result_dir + '*.png', test_high, use_GT_mean=True)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.4f}, Loss: {avg_train_loss:.6f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')

        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            model_path = "/content/drive/MyDrive/Att-UNet/best_psnr_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with PSNR: {best_psnr:.4f}')

        # add SSIM
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            model_path = "/content/drive/MyDrive/Att-UNet/best_ssim_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with SSIM: {best_ssim:.4f}')
        
        # add LPIPS
        if avg_lpips < best_lpips:
            best_lpips = avg_lpips
            model_path = "/content/drive/MyDrive/Att-UNet/best_lpips_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f'Saving model with LPIPS: {best_lpips:.4f}')
        
        # write log
        now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        file_path = "/content/drive/MyDrive/Att-UNet/results/metrics.md"
        file_exists = os.path.exists(file_path)

        with open(file_path, "a") as f:
            if not file_exists:
                f.write("|   Timestemp   |   Epoch   |    LR    |   Loss   |   PSNR   |   SSIM   |   LPIPS   |\n")
                f.write("|---------------|-----------|----------|----------|----------|----------|-----------|\n")
            
            f.write(f"|   {now}   | {epoch + 1} | {current_lr:.4f} | {avg_train_loss:.6f} |  {avg_psnr:.4f}  |  {avg_ssim:.4f}  |  {avg_lpips:.4f}  |\n")

if __name__ == '__main__':
    main()