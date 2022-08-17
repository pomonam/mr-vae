from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch


def load_cifar_data(split, batch_size, workers=4, data_path="../../logs/data"):
  normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                   std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
  if split == "train":
    transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ])
    dataset = datasets.CIFAR10(data_path, train=True, download=True,
                               transform=transform)

  elif split == "train_eval":
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize,
    ])
    dataset = datasets.CIFAR10(data_path, train=True, download=True,
                               transform=transform)

  elif split == "test":
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize,
    ])
    dataset = datasets.CIFAR10(data_path, train=False, download=True,
                               transform=transform)

  else:
    raise ValueError("Invalid split {:split}")

  is_train = split == "train"
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
  loader = load_cifar_data(split=split, batch_size=batch_size, data_path=data_path)

  for batch in loader:
    yield {"inputs": batch[0].to(device, non_blocking=True)}
