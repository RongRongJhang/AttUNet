from thop import profile
import torch
import time
from model import LaaFNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LaaFNet().to(device)  
input = torch.rand(1, 3, 256, 256).to(device)

torch.cuda.synchronize()
model.eval()
time_start = time.time()
_ = model(input)
time_end = time.time()
torch.cuda.synchronize()

time_sum = time_end - time_start
n_param = sum([p.nelement() for p in model.parameters()])  
macs, params = profile(model, inputs=(input,))

print(f"Time: {time_sum}")
print(f"Params(M): {(n_param/2**20)}")
print(f"FLOPs(G): {macs/(2**30)}")