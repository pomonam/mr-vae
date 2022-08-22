import torch
from torch import nn

from experiments.image.models import ResNetEncoder, ResNetDecoder
from src.criterions import kl_gaussian, log_sum_exp
import torch.nn.functional as F
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae
import math


class ImageModel(BaseVae):

    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        if self.decoder.require_inputs:
            reconstruct_error = self.decoder.reconstruct_error(x, z)
            outputs_dict = {
                "inputs": x,
                "mean": outputs_dict["mean"],
                "log_var": outputs_dict["log_var"],
                "reconstruct_error": reconstruct_error
            }
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
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + log_var).sum(-1)).mean()

        z_samples = self.sampler.sample(outputs_dict)
        z_samples = z_samples.unsqueeze(1)
        mu, logvar = mu.unsqueeze(0), log_var.unsqueeze(0)
        var = log_var.exp()
        dev = z_samples - mu
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + log_var.sum(-1))
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)
        return (neg_entropy - log_qz.mean(-1)).item()


def build_model(data_name, device):
    encoder = ResNetEncoder(data_name)
    if data_name == "celeba":
        sampler = IsotropicGaussianSampler(nh=4096, nz=64)
    else:
        sampler = IsotropicGaussianSampler(nh=4096, nz=64)
    decoder = ResNetDecoder(data_name)
    model = ImageModel(
        encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


class ImageCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict, beta: float = 1.0):
        if "reconstruct_error" in outputs_dict:
            log_likelihood = -outputs_dict["reconstruct_error"]
        else:
            log_likelihood = -F.mse_loss(outputs_dict["inputs"], outputs_dict["logits"])
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
        return ImageCriterion.forward(outputs_dict, beta=1.0)


def build_criterion(device):
    loss_fnc = ImageCriterion()
    return loss_fnc.to(device)
