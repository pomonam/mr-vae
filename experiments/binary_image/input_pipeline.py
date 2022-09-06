import os
import urllib

import os
import pickle as pkl
import numpy as np
from PIL import Image
import scipy.io
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.datasets import MNIST
from urllib.request import urlretrieve


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


def load_mnist_binarized(data_path):
  dataset = os.path.join(data_path, "mnist.gz")

  if not os.path.isfile(dataset):

    datafiles = {
      "train": "http://www.cs.toronto.edu/~larocheh/public/"
               "datasets/binarized_mnist/binarized_mnist_train.amat",
      "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
               "binarized_mnist/binarized_mnist_valid.amat",
      "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
              "binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
      print("Downloading %s data..." % (split))
      datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

    pkl.dump([datasplits['train'], datasplits['valid'], datasplits['test']], open(dataset, "wb"))

  x_train, x_valid, x_test = pkl.load(open(dataset, "rb"))
  return x_train, x_valid, x_test


def load_mnist_data(split, batch_size, workers=0, data_path="logs/data"):
  assert split in ["train", "train_eval", "test"]

  train_data, _, test_data = load_mnist_binarized(data_path)
  train_data = train_data.reshape(-1, 1, 28, 28).astype('float32')
  test_data = test_data.reshape(-1, 1, 28, 28).astype('float32')
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
  assert split in ["train", "train_eval", "test"]
  download_omniglot(data_path)
  dataset = scipy.io.loadmat(os.path.join(data_path, "chardata.mat"))

  is_train = split == "train"
  train_transform = transforms.Compose([
    transforms.ToTensor(),
    Binarize(),
  ])

  test_transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  if split == "train" or split == "train_eval":
    data = dataset["data"].astype("float32").reshape(
        (28, 28, -1)).transpose((2, 1, 0))

    if is_train:
      dataset = Omniglot(data, train_transform)
    else:
      np.random.seed(777)
      data = np.random.binomial(1, data).astype("float32")
      dataset = Omniglot(data, test_transform)

  else:
    data = dataset["testdata"].astype("float32").reshape(
        (28, 28, -1)).transpose((2, 1, 0))

    np.random.seed(777)
    data = np.random.binomial(1, data).astype("float32")
    dataset = Omniglot(data, test_transform)

  loader = torch.utils.data.DataLoader(
      dataset,
      pin_memory=True,
      batch_size=batch_size,
      shuffle=True,
      num_workers=workers,
      drop_last=is_train,
      sampler=None)

  return loader
