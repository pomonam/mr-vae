import unittest

import torch

from experiments.binary_image.baseline_train import build_model
from experiments.binary_image.hyper_train import build_model as hyper_build_model
from src.config import HyperConfig
from tests.utils import summary


class TestBinaryImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.conv_model = build_model("conv", "conv", device)
    self.resnet_model = build_model("resnet", "resnet", device)

    cfg = HyperConfig(None)
    cfg.shared_preprocess = True
    cfg.param_type = "pre_bn"
    cfg.layer_type = "sig_gate"
    cfg.block_type = "mlp"
    cfg.apply_zero_init = True
    cfg.apply_norm_layers = False
    cfg.include_output_stem = False

    self.hyper_conv_model = hyper_build_model("conv", "conv", cfg, device)
    self.hyper_resnet_model = hyper_build_model("resnet", "resnet", cfg, device)

  def test_summary(self):
    print(summary(self.conv_model, (1, 28, 28), device="cpu"))
    print(summary(self.resnet_model, (1, 28, 28), device="cpu"))
    # print(summary(self.hyper_resnet_model, (1, 28, 28), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 1, 28, 28)
    outputs_dict = self.conv_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
    outputs_dict = self.resnet_model(x)
    print("Model Output size:", outputs_dict["reconstruction"].shape)
