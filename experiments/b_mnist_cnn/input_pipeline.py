from experiments.b_mnist_mlp.input_pipeline import load_data


def build_input_queue(split, batch_size, device, data_path="../../logs/data"):
    loader = load_data(split=split, batch_size=batch_size, data_path=data_path)

    for batch in loader:
        yield {
            "inputs": batch["inputs"].tile(1, 3, 1, 1).to(device, non_blocking=True),
            "targets": batch["targets"].to(device, non_blocking=True),
            "index": batch["index"]
        }
