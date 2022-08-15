import os
import urllib.request

import h5py
import numpy as np
from PIL import Image
import scipy.io
from scipy.io import loadmat
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data as data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision import transforms


class omniglot(data.Dataset):
    """ omniglot dataset """
    url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

    def __init__(self, root, train=True, transform=None, download=False):
        # we ignore transform.
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 28)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        img = torch.bernoulli(img)  # stochastically binarize
        return img, torch.tensor(-1)  # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):

        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28),
                                                      order='fortran')

        omni_raw = scipy.io.loadmat(os.path.join(self.root, 'chardata.mat'))
        data_str = 'data' if train else 'testdata'
        data = reshape_data(omni_raw[data_str].T.astype('float32'))
        return data

    def get_mean_img(self):
        return self.data.mean(0)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading from {}...'.format(self.url))
        local_filename = os.path.join(self.root, 'chardata.mat')
        urllib.request.urlretrieve(self.url, local_filename)
        print('Saved to {}'.format(local_filename))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'chardata.mat'))


def load_data(split, batch_size, workers=0, data_path="../../logs/data"):
    file_name = os.path.join(data_path, "binary_mnist.h5")
    # if not os.path.exists(file_name):
    #   download_binary_mnist(file_name)

    train_data, valid_data, test_data = load_binary_mnist(file_name)

    if split == "train" or split == "train_eval":
        dataset = train_data

    elif split == "valid":
        dataset = valid_data

    elif split == "test":
        dataset = test_data

    elif split == "analytical":
        dataset = train_data

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
    loader = load_data(split=split, batch_size=batch_size, data_path=data_path)

    for batch in loader:
        yield {
            "inputs":
                batch.view(batch.shape[0], 1, 28,
                           28).to(device, non_blocking=True),
            "targets":
                batch.view(batch.shape[0], 1, 28,
                           28).to(device, non_blocking=True),
        }


if __name__ == "__main__":
    load_omniglot()
