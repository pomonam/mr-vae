import os
import urllib.request

import h5py
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed


def parse_binary_mnist(data_dir):

    def lines_to_np_array(ls):
        return np.array([[int(i) for i in line.split()] for line in ls])

    with open(os.path.join(data_dir, "binarized_mnist_train.amat")) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(data_dir, "binarized_mnist_valid.amat")) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype("float32")
    with open(os.path.join(data_dir, "binarized_mnist_test.amat")) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype("float32")
    return train_data, validation_data, test_data


def download_binary_mnist(data_path):
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    subdatasets = ["train", "valid", "test"]
    for subdataset in subdatasets:
        fn = "binarized_mnist_{}.amat".format(subdataset)
        url = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat".format(
            subdataset)
        local_filename = os.path.join(data_path, fn)
        urllib.request.urlretrieve(url, local_filename)

    train, validation, test = parse_binary_mnist(data_path)

    data_dict = {"train": train, "valid": validation, "test": test}
    file_name = os.path.join(data_path, "binary_mnist.h5")
    f = h5py.File(file_name, "w")
    f.create_dataset("train", data=data_dict["train"])
    f.create_dataset("valid", data=data_dict["valid"])
    f.create_dataset("test", data=data_dict["test"])
    f.close()


def load_binary_mnist(file_name):
    f = h5py.File(file_name, "r")
    x_train = f["train"][::]
    x_val = f["valid"][::]
    x_test = f["test"][::]
    return x_train, x_val, x_test


def load_data(split, batch_size, workers=0, data_path="logs/data"):
    file_name = os.path.join(data_path, "binary_mnist.h5")
    if not os.path.exists(file_name):
        download_binary_mnist(data_path)

    train_data, valid_data, test_data = load_binary_mnist(file_name)

    if split == "train" or split == "train_eval":
        dataset = train_data

    elif split == "valid":
        dataset = valid_data

    elif split == "test":
        dataset = test_data

    else:
        raise ValueError("Invalid split {:split}")

    is_train = split == "train"
    loader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=is_train,
        sampler=None)

    return loader


def build_input_queue(loader, device):
    for batch in loader:
        yield {"data": batch.view(batch.shape[0], 1, 28, 28).to(device, non_blocking=True)}
