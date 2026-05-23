import unittest

import torch

from experiments.image.baseline_train import build_model
from experiments.image.hyper_train import build_model as hyper_build_model
from tests.utils import summary


class TestCelebAImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model("celeba", "conv", device)
    self.resnet_model = build_model("celeba", "resnet", device)
    self.hyper_conv_model = hyper_build_model("celeba", "conv", device)
    self.hyper_resnet_model = hyper_build_model("celeba", "resnet", device)

  def test_summary(self):
    print(summary(self.resnet_model, (3, 64, 64), device="cpu"))
    print(summary(self.hyper_resnet_model, (3, 64, 64), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 3, 64, 64)
    print("Model Output size:", self.conv_model(x)["reconstruction"].shape)
    print("Model Output size:", self.resnet_model(x)["reconstruction"].shape)
    print("Model Output size:", self.hyper_conv_model(x)["reconstruction"].shape)
    print("Model Output size:", self.hyper_resnet_model(x)["reconstruction"].shape)
