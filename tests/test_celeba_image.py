import unittest

import torch

from experiments.image.baseline_train import build_model
from experiments.image.hyper_train import \
    build_model as hyper_build_model
from src.config import HyperConfig
from tests.utils import summary


class TestCelebAImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model("celeba", "conv", device)
    self.resnet_model = build_model("celeba", "resnet", device)

    cfg = HyperConfig(None)
    cfg.initialize_default_config()
    self.hyper_conv_model = hyper_build_model("celeba", "conv", cfg, device)
    self.hyper_resnet_model = hyper_build_model("celeba", "resnet", cfg, device)

  def test_summary(self):
    print(summary(self.resnet_model, (3, 64, 64), device="cpu"))
    print(summary(self.hyper_resnet_model, (3, 64, 64), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 3, 64, 64)
    outputs_dict = self.conv_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
    outputs_dict = self.resnet_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
    outputs_dict = self.hyper_conv_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
    outputs_dict = self.hyper_resnet_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
