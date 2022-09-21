import torch

t1 = torch.randn((2,3))
t2 = torch.randn((3,4))

print(t1 @ t2)
print((t1@t2).shape)