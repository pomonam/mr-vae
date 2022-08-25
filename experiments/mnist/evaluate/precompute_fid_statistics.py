import os

import torch

from experiments.mnist.input_pipeline import load_data
from src.fid.fid_score import compute_statistics_of_generator
from src.fid.fid_score import save_statistics
from src.fid.inception import InceptionV3


def main():
    device = torch.device('cuda')
    dims = 2048
    batch_size = 128
    fid_dir = '/home/baejuhan/codes/hyper-vae/experiments/mnist/checkpoints'
    train_queue = load_data(
        "train_eval", batch_size, data_path="../../../logs/data")
    valid_queue = load_data(
        "test", batch_size, data_path="../../../logs/data")
    print('len train queue',
          len(train_queue),
          'len val queue',
          len(valid_queue),
          'batch size',
          batch_size)
    # if args.dataset in {'celeba_256', 'omniglot'}:
    #     train_queue = chain(train_queue, valid_queue)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=fid_dir).to(device)
    m, s = compute_statistics_of_generator(train_queue, model, batch_size, dims, device, 50000)
    file_path = os.path.join(fid_dir, 'mnist.npz')
    print('saving fid stats at %s' % file_path)
    save_statistics(file_path, m, s)


if __name__ == '__main__':
    main()
