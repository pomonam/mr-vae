import texar.torch as tx


def load_data(name, split, batch_size, data_path="../../logs/text_data", device=None):
  if name == "yahoo":
    return load_yahoo_data(name, split, batch_size, data_path, device)
  elif name == "ptb":
    return load_ptb_data(name, split, batch_size, data_path, device)
  else:
    raise NotImplementedError


def load_yahoo_data(name, split, batch_size, data_path="../../logs/text_data", device=None):
  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/data/yahoo/yahoo.train.txt',
      "vocab_file": '../../logs/data/data/yahoo/vocab.txt',
    }
  }
  train_data = tx.data.MonoTextData(train_data_hparams, device=device)
  vocab = train_data.vocab

  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/data/yahoo/yahoo.valid.txt',
      "vocab_file": '../../logs/data/data/yahoo/vocab.txt',
    }
  }
  valid_data = tx.data.MonoTextData(train_data_hparams, device=device)

  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/data/yahoo/yahoo.test.txt',
      "vocab_file": '../../logs/data/data/yahoo/vocab.txt',
    }
  }
  test_data = tx.data.MonoTextData(train_data_hparams, device=device)

  iterator = tx.data.DataIterator(
    {"train": train_data, "valid": valid_data, "test": test_data})

  return train_data, iterator, vocab


def load_ptb_data(name, split, batch_size, data_path="../../logs/text_data", device=None):
  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/simple-examples/data/ptb.train.txt',
      "vocab_file": '../../logs/data/simple-examples/data/vocab.txt',
    }
  }
  train_data = tx.data.MonoTextData(train_data_hparams, device=device)
  vocab = train_data.vocab

  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/simple-examples/data/ptb.valid.txt',
      "vocab_file": '../../logs/data/simple-examples/data/vocab.txt',
    }
  }
  valid_data = tx.data.MonoTextData(train_data_hparams, device=device)

  train_data_hparams = {
    "num_epochs": 1,
    "batch_size": batch_size,
    "seed": 123,
    "dataset": {
      "files": '../../logs/data/simple-examples/data/ptb.test.txt',
      "vocab_file": '../../logs/data/simple-examples/data/vocab.txt',
    }
  }
  test_data = tx.data.MonoTextData(train_data_hparams, device=device)

  iterator = tx.data.DataIterator(
    {"train": train_data, "valid": valid_data, "test": test_data})

  return train_data, iterator, vocab
