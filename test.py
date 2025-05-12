import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LaaFNet
from dataloader import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
import lpips
from datetime import datetime
from measure import metrics
from torch.utils.data import DataLoader
from measure_niqe_bris import metrics as metrics_niqu

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
    # Paths and device setup
    # test_low = 'data/LOLv1/Test/input'
    # test_high = 'data/LOLv1/Test/target'

    # test_low = 'data/LOLv2/Real_captured/Test/Low'
    # test_high = 'data/LOLv2/Real_captured/Test/Normal'

    # test_low = 'data/LOLv2/Synthetic/Test/Low'
    # test_high = 'data/LOLv2/Synthetic/Test/Normal'

    test_low = 'data/LIME/*.bmp'

    weights_path = '/content/drive/MyDrive/Att-UNet/best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = test_low.split('/')[1]
    result_dir = '/content/drive/MyDrive/Att-UNet/results/testing/output/'

    # _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    test_loader = DataLoader(test_low, num_workers=4, batch_size=1, shuffle=False)

    print(f'Test loader: {len(test_loader)}')

    model = LaaFNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    validate(model, test_loader, device, result_dir)

    # avg_psnr, avg_ssim, avg_lpips = metrics(result_dir + '*.png', test_high, use_GT_mean=True)
    # print(f'Validation PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}')
    avg_niqe, avg_brisque = metrics_niqu(result_dir)
    print(f'Validation NIQE: {avg_niqe:.4f}, BRISQUE: {avg_brisque:.4f}')

    # write log
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    file_path = "/content/drive/MyDrive/Att-UNet/results/testing/metrics.md"
    file_exists = os.path.exists(file_path)

    # with open(file_path, "a") as f:
    #     if not file_exists:
    #         f.write("| Timestemp | PSNR | SSIM | LPIPS |\n")
    #         f.write("|-----------|------|------|-------|\n")
        
    #     f.write(f"| {now} | {avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")
    with open(file_path, "a") as f:
        if not file_exists:
            f.write("| Timestemp |   NIQE   |   BRISQUE   |\n")
            f.write("|-----------|----------|-------------|\n")
        
        f.write(f"| {now} | {avg_niqe:.4f} | {avg_brisque:.4f} |\n")

if __name__ == '__main__':
    main()
