from experiments.b_mnist_mlp.input_pipeline import build_input_queue as b_mnist_mlp_build

import torch


def main():
    b_mnist_mlp_build("train", 1, device=torch.device("cpu"), data_path="logs/data")
    b_mnist_mlp_build("test", 1, device=torch.device("cpu"), data_path="logs/data")


if __name__ == "__main__":
    main()
