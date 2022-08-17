import os


def main():
    from experiments.b_mnist.input_pipeline import load_data
    if not os.path.isdir("logs/data"):
        os.mkdir("logs/data")
    load_data("train", 1, data_path="logs/data")
    load_data("test", 1, data_path="logs/data")

    from experiments.cifar.input_pipeline import load_cifar_data
    load_cifar_data("train", 1, data_path="logs/data")
    load_cifar_data("test", 1, data_path="logs/data")


if __name__ == "__main__":
    main()
