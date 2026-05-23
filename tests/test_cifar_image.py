import unittest

import torch

from experiments.image.baseline_train import build_model
from experiments.image.hyper_train import build_model as hyper_build_model
from tests.utils import summary


class TestCifarImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model("cifar", "conv", device)
    self.resnet_model = build_model("cifar", "resnet", device)
    self.hyper_conv_model = hyper_build_model("cifar", "conv", device)
    self.hyper_resnet_model = hyper_build_model("cifar", "resnet", device)

  def test_summary(self):
    print(summary(self.resnet_model, (3, 32, 32), device="cpu"))
    print(summary(self.hyper_resnet_model, (3, 32, 32), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 3, 32, 32)
    print("Model Output size:", self.conv_model(x)["reconstruction"].shape)
    print("Model Output size:", self.resnet_model(x)["reconstruction"].shape)
    print("Model Output size:", self.hyper_conv_model(x)["reconstruction"].shape)
    print("Model Output size:", self.hyper_resnet_model(x)["reconstruction"].shape)
