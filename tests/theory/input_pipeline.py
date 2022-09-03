from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch


def load_mnist_data(split, batch_size, workers=1, data_path="../../logs/data"):
  if split == "train" or split == "analytical" or split == "train_eval":
    dataset = datasets.MNIST(data_path, train=True, download=True,
                               transform=transforms.ToTensor())
    is_train = True

  elif split == "test":
    dataset = datasets.MNIST(data_path, train=False, download=True,
                               transform=transforms.ToTensor())
    is_train = False

  else:
    raise ValueError("Invalid split {:split}")

  loader = torch.utils.data.DataLoader(
    dataset,
    pin_memory=True,
    batch_size=batch_size,
    shuffle=is_train,
    num_workers=workers,
    drop_last=is_train,
    sampler=None)

  return loader


def build_input_queue(split, batch_size, device, data_path="../../logs/data"):
  loader = load_mnist_data(split=split, batch_size=batch_size, data_path=data_path)

  for batch in loader:
    yield {"inputs": batch[0].to(device, non_blocking=True)}

