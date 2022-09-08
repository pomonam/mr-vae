import os

import texar.torch as tx


def prepare_data(data_name):
  if data_name == "ptb":
    data_path = "logs/data"
    train_path = os.path.join(data_path, "ptb.train.txt")
    if not os.path.exists(train_path):
      url = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
      tx.data.maybe_download(url, data_path, extract=True)

    train_path = os.path.join(data_path, "simple-examples/data/ptb.train.txt")
    vocab_path = os.path.join(data_path, "simple-examples/data/vocab.txt")
    word_to_id = tx.data.make_vocab(
      train_path, return_type="dict")

    with open(vocab_path, 'w') as fvocab:
      for word in word_to_id:
        fvocab.write("%s\n" % word)

  elif data_name == "yahoo":
    data_path = "logs/data"
    train_path = os.path.join(data_path, "yahoo.train.txt")
    if not os.path.exists(train_path):
      url = 'https://drive.google.com/file/d/' \
            '13IsiffVjcQ-wrrbBGMwiG3sYf-DFxtXH/view?usp=sharing'
      tx.data.maybe_download(url, path=data_path, filenames='yahoo.zip',
                             extract=True)
  else:
    raise ValueError('Unknown data: {}'.format(data_name))


def main():
  # Binary images ...
  from experiments.binary_image.input_pipeline import load_mnist_data
  from experiments.binary_image.input_pipeline import load_omniglot_data
  if not os.path.isdir("logs/data"):
    os.mkdir("logs/data")
  load_mnist_data("train", 1, data_path="logs/data")
  load_mnist_data("test", 1, data_path="logs/data")
  load_omniglot_data("train", 1, data_path="logs/data")
  load_omniglot_data("test", 1, data_path="logs/data")

  # Images ...
  from experiments.image.input_pipeline import load_data
  load_data("cifar", "train", 1, data_path="logs/data")
  load_data("cifar", "test", 1, data_path="logs/data")
  load_data("svhn", "train", 1, data_path="logs/data")
  load_data("svhn", "test", 1, data_path="logs/data")
  # Note that CelebA dataset should already be here...!

  # Texts ...
  prepare_data("ptb")
  prepare_data("yahoo")


if __name__ == "__main__":
  main()
