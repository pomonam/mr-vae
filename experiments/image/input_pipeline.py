import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
from torchvision.datasets import CelebA

import torchvision.transforms as transforms


def load_data(data_name, split, batch_size, workers=0, data_path="../../logs/data"):
    if data_name == "cifar":
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_data = CIFAR10(data_path,
                            train=True,
                            download=True,
                            transform=transform_train)
        test_data = CIFAR10(data_path,
                           train=False,
                           download=True,
                           transform=transform_test)

    elif data_name == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([x / 255.0 for x in[109.9, 109.7, 113.8]],
                                 [x / 255.0 for x in [50.1, 50.6, 50.8]])
        ])
        train_data = SVHN(data_path,
                             split='train',
                             download=True,
                             transform=transform)
        test_data = SVHN(data_path,
                        split='test',
                        download=True,
                            transform=transform)

    elif data_name == "celeba":
        train_data = test_data = None

    else:
        raise NotImplementedError(
            "Invalid dataset {} provided.".format(data_name))

    is_train = split == "train"
    loader = torch.utils.data.DataLoader(train_data if split in ["train", "train_eval"] else test_data,
                                         pin_memory=True,
                                         batch_size=batch_size,
                                         shuffle=is_train,
                                         num_workers=workers,
                                         drop_last=is_train,
                                         sampler=None)

    return loader


def build_input_queue(data_name, split, batch_size, device, data_path="../../logs/data"):
    loader = load_data(data_name=data_name, split=split, batch_size=batch_size, data_path=data_path)

    for batch in loader:
        yield {"inputs": batch.to(device, non_blocking=True)}
