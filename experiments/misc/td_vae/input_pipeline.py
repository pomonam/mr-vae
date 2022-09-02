import os
import urllib

from PIL import Image
import scipy.io
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


class DisentangledSpritesDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    self.filepath = f'{self.root_dir}/{self.filename}'
    dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

    # print('Keys in the dataset:', dataset_zip.keys())
    self.imgs = dataset_zip['imgs']
    self.latents_values = dataset_zip['latents_values']
    self.latents_classes = dataset_zip['latents_classes']
    self.metadata = dataset_zip['metadata'][()]

    # print('Metadata: \n', self.metadata)
    self.transform = transform

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    sample = self.imgs[idx].astype(np.float32)
    # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    if self.transform:
      sample = self.transform(sample)
    return sample, []


def load_data(split, batch_size, workers=0, data_path="../../logs/"):
  dataset = DisentangledSpritesDataset(data_path, transform=transforms.ToTensor())

  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  val_split = 0.9
  _split = int(np.floor(val_split * dataset_size))
  np.random.seed(42)
  np.random.shuffle(indices)
  train_indices, val_indices = indices[_split:], indices[:_split]

  train_sampler = SubsetRandomSampler(train_indices)
  val_sampler = SubsetRandomSampler(val_indices)

  is_train = split == "train"
  loader = torch.utils.data.DataLoader(
      dataset,
      pin_memory=True,
      batch_size=batch_size,
      # shuffle=True,
      num_workers=workers,
      drop_last=is_train,
      sampler=train_sampler if split in ["train", "train_eval"] else val_sampler)

  return loader
