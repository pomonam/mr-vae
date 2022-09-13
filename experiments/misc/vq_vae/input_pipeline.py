import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets import MNIST


def load_mnist_data(split, batch_size, workers=0, data_path="logs/data"):
  # Fresh MNIST loader here for MSE loss.
  train_transform = transforms.Compose([
      transforms.ToTensor(),
  ])
  test_transform = transforms.Compose([
      transforms.ToTensor(),
  ])

  train_data = MNIST(
      root=data_path,
      train=True,
      download=True,
      transform=train_transform if split == "train" else test_transform)

  test_data = MNIST(
      root=data_path, train=False, download=True, transform=test_transform)

  is_train = split == "train"
  loader = torch.utils.data.DataLoader(
      train_data if split in ["train", "train_eval"] else test_data,
      pin_memory=True,
      batch_size=batch_size,
      shuffle=True,
      num_workers=workers,
      drop_last=is_train,
      sampler=None)

  return loader
