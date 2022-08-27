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
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
  https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
  """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def load_data(data_name,
              split,
              batch_size,
              workers=4,
              data_path="../../logs/data"):
    if data_name == "cifar":
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_data = CIFAR10(
            data_path, train=True, download=True, transform=train_transform)
        test_data = CIFAR10(
            data_path, train=False, download=True, transform=test_transform)

    elif data_name == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = SVHN(
            data_path, split='train', download=True, transform=transform)
        test_data = SVHN(
            data_path, split='test', download=True, transform=transform)

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
        train_data = CelebA(
            "/scratch/ssd002/datasets/celeba_pytorch",
            split='train',
            download=True,
            transform=train_transform)
        test_data = CelebA(
            "/scratch/ssd002/datasets/celeba_pytorch",
            split='test',
            download=True,
            transform=test_transform)

    else:
        raise NotImplementedError(
            "Invalid dataset {} provided.".format(data_name))

    is_train = split == "train"
    loader = torch.utils.data.DataLoader(
        train_data if split in ["train", "train_eval"] else test_data,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        drop_last=is_train,
        sampler=None)

    return loader
