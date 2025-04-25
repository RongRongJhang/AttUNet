from thop import profile
import torch
import time
from model4_modify9 import LYT

# 設定裝置
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LYT().to(device)  
input = torch.rand(1, 3, 256, 256).to(device)  
torch.mps.synchronize()
model.eval()
time_start = time.time()
_ = model(input)
time_end = time.time()
torch.mps.synchronize()
time_sum = time_end - time_start
print(f"Time: {time_sum}")
n_param = sum([p.nelement() for p in model.parameters()])  
n_paras = f"n_paras: {(n_param/2**20)}M\n"
print(n_paras)
macs, params = profile(model, inputs=(input,)) 
print(f'FLOPs:{macs/(2**30)}G')