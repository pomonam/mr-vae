import torch
from torch import nn

from experiments.b_mnist.models.decoders import CNNDecoder
from experiments.b_mnist.models.decoders import MLPDecoder
from experiments.b_mnist.models.decoders import PixelCNNDecoder
from experiments.b_mnist.models.encoders import CNNEncoder
from experiments.b_mnist.models.encoders import MLPEncoder
from experiments.b_mnist.models.encoders import ResNetEncoder
from experiments.b_mnist.models.hyper_decoders import HyperCNNDecoder
from experiments.b_mnist.models.hyper_decoders import HyperMLPDecoder
from experiments.b_mnist.models.hyper_encoders import HyperCNNEncoder
from experiments.b_mnist.models.hyper_encoders import HyperMLPEncoder
from src.criterions import binary_cross_entropy
from src.criterions import kl_gaussian, log_sum_exp
from src.hyper.models import HyperIsotropicGaussianSampler
from src.hyper.vae import HyperVae
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae
import math


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

    def calc_mi(self, x):
        outputs_dict = self.encode(x)
        mu, log_var = outputs_dict["mean"], outputs_dict["log_var"]
        x_batch, nz = mu.size()
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + log_var).sum(-1)).mean()

        z_samples = self.sampler.sample(outputs_dict)
        z_samples = z_samples.unsqueeze(1)
        mu, logvar = mu.unsqueeze(0), log_var.unsqueeze(0)
        var = log_var.exp()
        dev = z_samples - mu
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + log_var.sum(-1))
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()


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
    else:
        raise ValueError()

    sampler = IsotropicGaussianSampler(nh=256, nz=64)

    if decoder_name == "mlp":
        decoder = MLPDecoder()
    elif decoder_name == "cnn":
        decoder = CNNDecoder()
    elif decoder_name == "pixelcnn":
        decoder = PixelCNNDecoder()
    else:
        raise ValueError()
    model = BinarizedMnistMlpModel(
        encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


def build_hyper_model(encoder_name, decoder_name, hyper_config, device):
    if encoder_name == "mlp":
        encoder = HyperMLPEncoder(hyper_config)
    elif encoder_name == "cnn":
        encoder = HyperCNNEncoder(hyper_config)
    elif encoder_name == "resnet":
        encoder = ResNetEncoder()
    else:
        raise ValueError()

    sampler = HyperIsotropicGaussianSampler(
        nh=256, nz=64, hyper_config=hyper_config)

    if decoder_name == "mlp":
        decoder = HyperMLPDecoder(hyper_config)
    elif decoder_name == "cnn":
        decoder = HyperCNNDecoder(hyper_config)
    elif decoder_name == "pixelcnn":
        decoder = PixelCNNDecoder()
    else:
        raise ValueError()
    model = HyperBinarizedMnistMlpModel(
        encoder=encoder,
        decoder=decoder,
        sampler=sampler,
        hyper_config=hyper_config)
    return model.to(device)


class BinarizedMnistMlpCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict, beta: torch.Tensor = 1.0):
        log_likelihood = -binary_cross_entropy(outputs_dict["inputs"],
                                               outputs_dict["logits"])
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


class HyperBinarizedMnistMlpCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict):
        log_likelihood = -binary_cross_entropy(outputs_dict["inputs"],
                                               outputs_dict["logits"])
        kl = kl_gaussian(outputs_dict["mean"], outputs_dict["log_var"].exp())

        if "beta" in outputs_dict:
            elbo = log_likelihood - outputs_dict["beta"].view(-1) * kl
        elif "alpha" in outputs_dict:
            alpha = outputs_dict["alpha"].view(-1)
            elbo = alpha * log_likelihood - (1 - alpha) * kl
        else:
            raise NotImplementedError

        elbo = -torch.mean(elbo)
        loss_dict = {
            "loss": elbo.item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return BinarizedMnistMlpCriterion.forward(outputs_dict)


def build_criterion(device):
    loss_fnc = BinarizedMnistMlpCriterion()
    return loss_fnc.to(device)


def build_hyper_criterion(device):
    loss_fnc = HyperBinarizedMnistMlpCriterion()
    return loss_fnc.to(device)
