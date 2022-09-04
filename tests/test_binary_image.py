import unittest

import torch

from experiments.binary_image.baseline_train import build_model
from tests.utils import summary


class TestBinaryImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model("conv", "conv", device)
    self.resnet_model = build_model("resnet", "resnet", device)

  def test_summary(self):
    print(summary(self.conv_model, (1, 28, 28), device="cpu"))
    print(summary(self.resnet_model, (1, 28, 28), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 1, 28, 28)
    outputs_dict = self.conv_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
    outputs_dict = self.resnet_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
