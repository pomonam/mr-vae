import os


def main():
    from experiments.binary_image.input_pipeline import load_mnist_data
    if not os.path.isdir("logs/data"):
        os.mkdir("logs/data")
    load_mnist_data("train", 1, data_path="logs/data")
    load_mnist_data("test", 1, data_path="logs/data")

    # from experiments.image.input_pipeline import load_data
    # load_data("cifar", "train", 1, data_path="logs/data")
    # load_data("cifar", "test", 1, data_path="logs/data")
    # load_data("svhn", "train", 1, data_path="logs/data")
    # load_data("svhn", "test", 1, data_path="logs/data")


if __name__ == "__main__":
    main()
