import torch

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch device:", torch.cuda.get_device_name(0))

x = torch.randn(3, 3).cuda()
y = x @ x
print("Torch works:", y)
