import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LYT
from dataloader import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
import lpips
from datetime import datetime
from measure import metrics

def validate(model, dataloader, device, result_dir):
    model.eval()
    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            # Save the output image
            save_image(output, os.path.join(result_dir, f'{idx}.png'))

def main():
    # Paths and device setup
    test_low = 'data/LOLv1/Test/input'
    test_high = 'data/LOLv1/Test/target'
    weights_path = '/content/drive/MyDrive/Att-UNet/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = test_low.split('/')[1]
    result_dir = '/content/drive/MyDrive/Att-UNet/results/testing/output/'

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    print(f'Test loader: {len(test_loader)}')

    model = LYT().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    validate(model, test_loader, device, result_dir)
    avg_psnr, avg_ssim, avg_lpips = metrics(result_dir + '*.png', test_high, use_GT_mean=True)

    print(f'Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')

    # write log
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    file_path = "/content/drive/MyDrive/Att-UNet/results/testing/metrics.md"
    file_exists = os.path.exists(file_path)

    with open(file_path, "a") as f:
        if not file_exists:
            f.write("| Timestemp | PSNR | SSIM | LPIPS |\n")
            f.write("|-----------|------|------|-------|\n")
        
        f.write(f"| {now} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")

if __name__ == '__main__':
    main()
