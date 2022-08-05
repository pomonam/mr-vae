import torch

arr = torch.ones(2, 3, 4)
print(tuple((-1,)) + tuple((4,3)))
arr = arr.reshape(tuple((-1,)) + tuple((4,3)))
print(arr.shape)