import torch
from torch import nn

from experiments.cifar.models.decoders import ResNetDecoder
from experiments.cifar.models.encoders import ResNetEncoder
from src.hyper.vae import HyperVae
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae
from src.criterions import kl_gaussian
from torch.nn import functional as F


class CifarModel(BaseVae):

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


def build_model(device):
    encoder = ResNetEncoder()
    sampler = IsotropicGaussianSampler(nh=128 * 8 * 8, nz=64)
    decoder = ResNetDecoder()
    model = CifarModel(
        encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


class CifarCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict, beta: float = 1.0):
        log_likelihood = -F.mse_loss(outputs_dict["inputs"],
                                     outputs_dict["logits"],
                                     reduction="none").sum(dim=(1, 2, 3))
        kl = kl_gaussian(outputs_dict["mean"], outputs_dict["log_var"].exp())
        elbo = log_likelihood - beta * kl
        elbo = -torch.mean(elbo)
        loss_dict = {
            "loss": elbo.item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return CifarCriterion.forward(outputs_dict, beta=1.0)


def build_criterion(device):
    loss_fnc = CifarCriterion()
    return loss_fnc.to(device)
