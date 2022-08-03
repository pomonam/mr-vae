import unittest

import torch
from torchsummary import summary

from experiments.b_mnist_mlp import model_pipeline


class TestBMnistMlp(unittest.TestCase):
    def setUp(self) -> None:
        device = torch.device("cpu")
        self.model = model_pipeline.build_model(device)

    def test_summary(self):
        print(summary(self.model, (1, 784), device="cpu"))

    def test_forward(self):
        x = torch.randn(1, 784)
        outputs_dict = self.model(x)
        print("Model Output size:", outputs_dict["logits"])
