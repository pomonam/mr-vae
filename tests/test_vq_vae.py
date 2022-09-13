import unittest

import torch

from experiments.misc.vq_vae.baseline_train import build_model
from experiments.misc.vq_vae.hyper_train import \
    build_model as hyper_build_model
from src.config import HyperConfig
from tests.utils import summary


class TestVQImage(unittest.TestCase):

  def setUp(self) -> None:
    device = torch.device("cpu")
    self.mnist_model = build_model("mnist", 1., device)
    self.celeb_model = build_model("celeba", 1., device)

    cfg = HyperConfig(None)
    cfg.initialize_default_config()
    self.hyper_mnist_model = hyper_build_model("mnist", cfg, device)
    self.hyper_celeb_model = hyper_build_model("celeba", cfg, device)

  def test_summary(self):
    print(summary(self.mnist_model, (1, 28, 28), device="cpu"))
    print(summary(self.hyper_mnist_model, (1, 28, 28), device="cpu"))

  def test_forward(self):
    x = torch.randn(2, 1, 28, 28)
    outputs_dict = self.mnist_model(x)
    print("Model Output size:", outputs_dict["recon_x"].shape)
    outputs_dict = self.hyper_mnist_model(x)
    print("Model Output size:", outputs_dict["recon_x"].shape)
    x = torch.randn(2, 3, 64, 64)
    outputs_dict = self.celeb_model(x)
    print("Model Output size:", outputs_dict["recon_x"].shape)
    outputs_dict = self.hyper_celeb_model(x)
    print("Model Output size:", outputs_dict["recon_x"].shape)
