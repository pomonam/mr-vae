import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision.datasets import CelebA
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
import torchvision.transforms as transforms


class CropCelebA64(object):

  def __call__(self, pic):
    new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
    return new_pic

  def __repr__(self):
    return self.__class__.__name__ + "()"


def load_data(data_name,
              split,
              batch_size,
              workers=0,
              data_path="../../logs/data"):
  if data_name == "cifar":
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_data = CIFAR10(
        data_path, train=True, download=True, transform=train_transform)

    # num_train = 45000
    # train_data.data = train_data.data[:num_train, :, :, :]
    # train_data.targets = train_data.targets[:num_train]
    #
    # valid_data = CIFAR10(
    #     data_path, train=True, download=True, transform=test_transform)
    # valid_data.data = valid_data.data[num_train:, :, :, :]
    # valid_data.targets = valid_data.targets[num_train:]

    test_data = CIFAR10(
        data_path, train=False, download=True, transform=test_transform)

  elif data_name == "svhn":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = SVHN(
        data_path, split="train", download=True, transform=transform)

    # train_ratio = 0.9
    # num_train = math.floor(len(train_data.data) * train_ratio)
    #
    # train_data.data = train_data.data[:num_train, :, :, :]
    # train_data.labels = train_data.labels[:num_train]
    #
    # valid_data = SVHN(
    #     data_path, split="train", download=True, transform=transform)
    # valid_data.data = valid_data.data[num_train:, :, :, :]
    # valid_data.labels = valid_data.labels[num_train:]

    test_data = SVHN(
        data_path, split="test", download=True, transform=transform)

  elif data_name == "celeba":
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    # Feed in default paths.
    train_data = CelebA(
        "/scratch/ssd002/datasets/celeba_pytorch",
        split="train",
        download=True,
        transform=train_transform)

    # valid_data = CelebA(
    #     "/scratch/ssd002/datasets/celeba_pytorch",
    #     split="valid",
    #     download=True,
    #     transform=test_transform)

    test_data = CelebA(
        "/scratch/ssd002/datasets/celeba_pytorch",
        split="test",
        download=True,
        transform=test_transform)

  else:
    raise NotImplementedError("Invalid dataset {} provided.".format(data_name))

  is_train = split == "train"
  if split in ["train", "train_eval"]:
    loader = torch.utils.data.DataLoader(
        train_data if split in ["train", "train_eval"] else test_data,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=is_train,
        sampler=None)
  # elif split == "valid":
  #   loader = torch.utils.data.DataLoader(
  #       valid_data,
  #       pin_memory=True,
  #       batch_size=batch_size,
  #       shuffle=True,
  #       num_workers=workers,
  #       drop_last=is_train,
  #       sampler=None)
  else:
    loader = torch.utils.data.DataLoader(
        test_data,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        sampler=None)

  return loader
