import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision import transforms


class DisentangledSpritesDataset(Dataset):

  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    self.filepath = f"{self.root_dir}/{self.filename}"
    dataset_zip = np.load(self.filepath, allow_pickle=True, encoding="bytes")

    self.imgs = dataset_zip["imgs"]
    self.latents_values = dataset_zip["latents_values"]
    self.latents_classes = dataset_zip["latents_classes"]
    self.metadata = dataset_zip["metadata"][()]

    self.transform = transform

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    sample = self.imgs[idx].astype(np.float32)
    if self.transform:
      sample = self.transform(sample)
    return sample, []


def load_data(split,
              batch_size,
              workers=0,
              data_path="../../logs/",
              shuffle=True):
  del split

  dataset = DisentangledSpritesDataset(
      data_path, transform=transforms.ToTensor())

  loader = torch.utils.data.DataLoader(
      dataset,
      pin_memory=True,
      batch_size=batch_size,
      shuffle=shuffle,
      num_workers=workers,
      drop_last=False,
      sampler=None)

  return loader
