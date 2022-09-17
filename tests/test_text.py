import unittest

import torch

from experiments.text.baseline_train import build_model
from experiments.text.hyper_train import build_model as hyper_build_model
from src.config import HyperConfig


class TestTextImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.lstm_model = build_model(10_000, "ptb", "lstm", device)
    self.trans_model = build_model(10_000, "ptb", "trans", device)

    cfg = HyperConfig(None)
    cfg.initialize_default_config()
    self.hyper_lstm_model = hyper_build_model(10_000,
                                              "ptb",
                                              "lstm",
                                              cfg,
                                              device)
    self.hyper_trans_model = hyper_build_model(10_000,
                                                "ptb",
                                                "trans",
                                                cfg,
                                                device)

  def test_summary(self):
    print(sum(p.numel() for p in self.lstm_model.parameters()))
    print(sum(p.numel() for p in self.trans_model.parameters()))
    print(sum(p.numel() for p in self.hyper_lstm_model.parameters()))
    print(sum(p.numel() for p in self.hyper_trans_model.parameters()))
