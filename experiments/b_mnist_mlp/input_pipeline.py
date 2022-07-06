import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.dataset
import torch.utils.data.distributed
from torchvision.datasets import MNIST
from torchvision import transforms


class DictMNIST(MNIST):
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return {"inputs": image, "targets": label, "index": index}


def load_data(split,
              batch_size,
              workers=0,
              data_path="../../logs/data"):
    transform_train = transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # Binarize the dataset
        lambda x: x > 0,
        lambda x: x.float()
    ])

    if split == "train" or split == "train_eval":
        dataset = DictMNIST(data_path,
                            train=True,
                            download=True,
                            transform=transform_train)
    elif split == "test":
        dataset = DictMNIST(data_path,
                            train=False,
                            download=True,
                            transform=transform_test)
    else:
        raise ValueError("Invalid split {:split}")

    is_train = split == "train"
    loader = torch.utils.data.DataLoader(dataset,
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
            "inputs": batch["inputs"].view(-1, 784).to(device, non_blocking=True),
            "targets": batch["targets"].to(device, non_blocking=True),
            "index": batch["index"]
        }
