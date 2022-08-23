import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.file_name = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.file_path = f'{self.data_path}/{self.file_name}'
        dataset_zip = np.load(self.file_path, allow_pickle=True, encoding='bytes')

        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        if self.transform:
            sample = self.transform(sample)
        return sample, []


def load_data(split, batch_size, workers=0, data_path="../../logs/dsprites/"):
    dataset = DisentangledSpritesDataset(data_path, transform=transforms.ToTensor())

    # Create data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = 0.2
    split = int(np.floor(val_split * dataset_size))
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    is_train = split == "train"
    if split in ["train", "train_eval"]:
        loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers,
            drop_last=is_train,
            sampler=train_sampler
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers,
            drop_last=is_train,
            sampler=val_sampler
        )

    return loader


def build_input_queue(split, batch_size, device, data_path="../../logs/dsprites"):
    loader = load_data(split=split, batch_size=batch_size, data_path=data_path)

    for batch in loader:
        yield {
            "inputs":
                batch[0].to(device, non_blocking=True),
        }

