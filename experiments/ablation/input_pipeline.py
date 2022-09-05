import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets import MNIST


def load_mnist_data(split, batch_size, workers=0, data_path="logs/data"):
  # Different to /binary_image, we don't binarize here.
  assert split in ["train", "train_eval", "test"]
  transform = transforms.Compose([
      transforms.ToTensor(),
  ])

  is_train = split == "train"

  train_data = MNIST(
      root=data_path,
      train=True,
      download=True,
      transform=transform)

  test_data = MNIST(
      root=data_path, train=False, download=True, transform=transform)

  loader = torch.utils.data.DataLoader(
      train_data if split in ["train", "train_eval"] else test_data,
      pin_memory=True,
      batch_size=batch_size,
      shuffle=True,
      num_workers=workers,
      drop_last=is_train,
      sampler=None)

  return loader
