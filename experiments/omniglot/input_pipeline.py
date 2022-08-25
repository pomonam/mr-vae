import os

import scipy.io

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed


def load_data(split, batch_size, workers=0, data_path="../../logs/", force_shuffle=False):
    dataset = scipy.io.loadmat(os.path.join(data_path, 'omniglot/chardata.mat'))

    if split == "train" or split == "train_eval":
        dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset["data"].transpose()))

    elif split == "test":
        dataset = torch.utils.data.TensorDataset(torch.Tensor(dataset["testdata"].transpose()))

    else:
        raise ValueError("Invalid split {:split}")

    is_train = split == "train"
    loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=is_train or force_shuffle,
        num_workers=workers,
        drop_last=is_train,
        sampler=None)

    return loader


def build_input_queue(split, batch_size, device, data_path="../../logs/"):
    loader = load_data(split=split, batch_size=batch_size, data_path=data_path)

    for batch in loader:
        yield {"inputs": batch[0].view(batch[0].shape[0], 1, 28, 28).to(device, non_blocking=True)}
