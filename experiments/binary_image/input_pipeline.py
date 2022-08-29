import os
import urllib

import scipy.io
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class Binarize(object):

  def __call__(self, pic):
    return torch.Tensor(pic.size()).bernoulli_(pic)

  def __repr__(self):
    return self.__class__.__name__ + "()"


class Omniglot(Dataset):

  def __init__(self, data, transform):
    self.data = data
    self.transform = transform

  def __getitem__(self, index):
    d = self.data[index]
    img = Image.fromarray(d)
    return self.transform(img), 0

  def __len__(self):
    return len(self.data)


def load_mnist_data(split, batch_size, workers=0, data_path="logs/data"):
  train_transform = transforms.Compose([
      transforms.ToTensor(),
      Binarize(),
  ])
  test_transform = transforms.Compose([
      transforms.ToTensor(),
  ])

  train_data = MNIST(
      root=data_path,
      train=True,
      download=True,
      transform=train_transform if split == "train" else test_transform)
  if split == "train_eval":
    train_data.data[train_data.data >= 127.5] = 255.
    train_data.data[train_data.data < 127.5] = 0.

  test_data = MNIST(
      root=data_path, train=False, download=True, transform=test_transform)
  test_data.data[test_data.data >= 127.5] = 255.
  test_data.data[test_data.data < 127.5] = 0.

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


def download_omniglot(data_dir):
  filename = 'chardata.mat'
  if not os.path.exists(data_dir):
    os.mkdir(data_dir)
  url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

  filepath = os.path.join(data_dir, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    print('Downloaded', filename)

  return


def load_omniglot_data(split, batch_size, workers=0, data_path="../../logs/"):
  download_omniglot(data_path)
  dataset = scipy.io.loadmat(os.path.join(data_path, "chardata.mat"))

  train_transform = transforms.Compose([
      transforms.ToTensor(),
      Binarize(),
  ])

  test_transform = transforms.Compose([
      transforms.ToTensor(),
  ])

  is_train = split == "train"
  if split == "train" or split == "train_eval":
    data = 255 * dataset["data"].astype("float32").reshape(
        (28, 28, -1)).transpose((2, 1, 0))
    data = data.astype("uint8")
    if is_train:
      dataset = Omniglot(data, train_transform)
    else:
      data[data >= 127.5] = 255.
      data[data < 127.5] = 0.
      dataset = Omniglot(data, test_transform)

  elif split == "test":
    data = 255 * dataset["testdata"].astype("float32").reshape(
        (28, 28, -1)).transpose((2, 1, 0))
    data = data.astype("uint8")
    data[data >= 127.5] = 255.
    data[data < 127.5] = 0.
    dataset = Omniglot(data, test_transform)

  else:
    raise ValueError("Invalid split {:split}")

  loader = torch.utils.data.DataLoader(
      dataset,
      pin_memory=True,
      batch_size=batch_size,
      shuffle=True,
      num_workers=workers,
      drop_last=is_train,
      sampler=None)

  return loader
