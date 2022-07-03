import torch


def main():
    from experiments.b_mnist_mlp.input_pipeline import load_data
    load_data("train", 1, data_path="logs/data")
    load_data("test", 1, data_path="logs/data")


if __name__ == "__main__":
    main()
