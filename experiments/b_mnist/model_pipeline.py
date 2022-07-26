import torch
from torch import nn

from experiments.b_mnist.models.decoders import CNNDecoder
from experiments.b_mnist.models.decoders import MLPDecoder
from experiments.b_mnist.models.decoders import PixelCNNDecoder
from experiments.b_mnist.models.decoders import LinearDecoder
from experiments.b_mnist.models.encoders import CNNEncoder
from experiments.b_mnist.models.encoders import MLPEncoder
from experiments.b_mnist.models.encoders import ResNetEncoder
from experiments.b_mnist.models.encoders import LinearEncoder
from src.criterions import binary_cross_entropy
from src.criterions import kl_gaussian
from src.hyper.vae import HyperVae
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae


class BinarizedMnistMlpModel(BaseVae):
    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        if self.decoder.require_inputs:
            logits = self.decoder.special_decode(z, x)
        else:
            logits = self.decode(z)

        outputs_dict = {
            "inputs": x,
            "mean": outputs_dict["mean"],
            "log_var": outputs_dict["log_var"],
            "logits": logits
        }
        return outputs_dict


class HyperBinarizedMnistMlpModel(HyperVae):
    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        logits = self.decode(z)
        outputs_dict = {
            "inputs": x,
            "mean": outputs_dict["mean"],
            "log_var": outputs_dict["log_var"],
            "logits": logits
        }
        return outputs_dict


def build_model(encoder_name, decoder_name, device):
    if encoder_name == "mlp":
        encoder = MLPEncoder()
    elif encoder_name == "cnn":
        encoder = CNNEncoder()
    elif encoder_name == "resnet":
        encoder = ResNetEncoder()
    elif encoder_name == "linear":
        encoder = LinearEncoder()
    else:
        raise ValueError()

    sampler = IsotropicGaussianSampler(nh=256, nz=64)

    if decoder_name == "mlp":
        decoder = MLPDecoder()
    elif decoder_name == "cnn":
        decoder = CNNDecoder()
    elif decoder_name == "pixelcnn":
        decoder = PixelCNNDecoder()
    elif decoder_name == "linear":
        decoder = LinearDecoder()
    else:
        raise ValueError()
    model = BinarizedMnistMlpModel(encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


def build_hyper_model(encoder_name, decoder_name, hyper_config, device):
    if encoder_name == "mlp":
        encoder = MLPEncoder()
    elif encoder_name == "cnn":
        encoder = CNNEncoder()
    elif encoder_name == "resnet":
        encoder = ResNetEncoder()
    elif encoder_name == "linear":
        encoder = LinearEncoder()
    else:
        raise ValueError()

    sampler = IsotropicGaussianSampler(nh=256, nz=64)

    if decoder_name == "mlp":
        decoder = MLPDecoder()
    elif decoder_name == "cnn":
        decoder = CNNDecoder()
    elif decoder_name == "pixelcnn":
        decoder = PixelCNNDecoder()
    elif decoder_name == "linear":
        decoder = LinearDecoder()
    else:
        raise ValueError()
    model = HyperBinarizedMnistMlpModel(encoder=encoder, decoder=decoder,
                                        sampler=sampler, hyper_config=hyper_config)
    return model.to(device)


class BinarizedMnistMlpCriterion(nn.Module):
    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict, beta: torch.Tensor = 1.0):
        log_likelihood = -binary_cross_entropy(outputs_dict["inputs"], outputs_dict["logits"])
        kl = kl_gaussian(outputs_dict["mean"], outputs_dict["log_var"].exp())
        if isinstance(beta, int) or isinstance(beta, float):
            if beta == -1:
                elbo = -kl
            else:
                elbo = log_likelihood - beta * kl
        else:
            elbo = log_likelihood - beta.view(-1) * kl
        elbo = -torch.mean(elbo)
        loss_dict = {
            "loss": elbo.item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return BinarizedMnistMlpCriterion.forward(outputs_dict, beta=1.0)


def build_criterion(device: torch.device) -> BinarizedMnistMlpCriterion:
    loss_fnc = BinarizedMnistMlpCriterion()
    return loss_fnc.to(device)
