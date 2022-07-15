import os


def main():
    from experiments.b_mnist.input_pipeline import load_data
    os.mkdir("logs/data")
    load_data("train", 1, data_path="logs/data")
    load_data("test", 1, data_path="logs/data")


if __name__ == "__main__":
    main()
