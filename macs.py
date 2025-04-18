import torch
from torchprofile import profile_macs
from model import LYT
from fvcore.nn import FlopCountAnalysis
from calflops import calculate_flops

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LYT().to(device)
input_tensor = torch.randn(1, 3, 256, 256).to(device) 

model.eval()  # 確保模型在評估模式
macs = profile_macs(model, input_tensor)
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

tflops = flops / (1024*1024*1024)

print(f"Model FLOPs (G): {tflops} G")
print(f"Model FLOPs (M): {tflops*1024} M")
print(f"Model MACs (G): {macs / (1024*1024*1024)} G")
print(f"Model params (M): {num_params / 1e6}")
print(f"Model params: {num_params}")



# def my_summary(test_model, H = 256, W = 256, C = 3, N = 1):
#     model = test_model
#     # print(model)
#     inputs = torch.randn((N, C, H, W)).to(device)
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])
#     print(f'FLOPs(G):{flops.total()}')
#     print(f'Macs(G):{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')

# my_summary(model)


# def cal_calflops(test_model):
#     model = test_model
#     input_shape = (1, 3, 256, 256)
#     flops, macs, params = calculate_flops(model=model, 
#                                       input_shape=input_shape,
#                                       output_as_string=True,
#                                       output_precision=4)
#     print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

# cal_calflops(model)

# import torch
# from torchvision.models import resnet152, resnet18
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# model = LYT()

# tensor = (torch.rand(1, 3, 256, 256),)

# #分析FLOPs
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())

# def print_model_parm_nums(model):
#     total = sum([param.nelement() for param in model.parameters()])
#     print('  + Number of params: %.2fM' % (total / 1e6))

# print_model_parm_nums(model)
