from experiments.binary_image.input_pipeline import load_mnist_data


def build_input_queue(split, batch_size, device, data_path="../../logs/data", dataset_size=-1):
  loader = load_mnist_data(split=split, batch_size=batch_size, dataset_size=dataset_size, data_path=data_path)

  for batch in loader:
    if isinstance(batch, list):
      yield {"inputs": batch[0].to(device, non_blocking=True)}
    else:
      yield {"inputs": batch.to(device, non_blocking=True)}

