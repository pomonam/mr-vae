import unittest

import torch

from experiments.text.baseline_train import build_model
from experiments.text.hyper_train import \
    build_model as hyper_build_model
from src.config import HyperConfig


class TestTextImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model(10_000, "ptb", "lstm", device)
    self.resnet_model = build_model(10_000, "ptb", "trans", device)

    cfg = HyperConfig(None)
    cfg.initialize_default_config()
    self.hyper_conv_model = hyper_build_model(10_000, "ptb", "lstm", cfg, device)
    self.hyper_resnet_model = hyper_build_model(10_000, "ptb", "lstm", cfg, device)

  def test_summary(self):
    pass
