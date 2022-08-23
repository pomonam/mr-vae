import torch
from torch import nn

from experiments.image.hyper_models import HyperResNetEncoder, HyperResNetDecoder
from src.criterions import kl_gaussian, log_sum_exp
import torch.nn.functional as F
from src.hyper.vae import HyperVae
from src.hyper.models import HyperIsotropicGaussianSampler
import math


class ImageModel(HyperVae):

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


def build_hyper_model(data_name, hyper_config, device):
    encoder = HyperResNetEncoder(data_name, hyper_config)
    if data_name == "celeba":
        sampler = HyperIsotropicGaussianSampler(nh=4096, nz=64, hyper_config=hyper_config)
    else:
        sampler = HyperIsotropicGaussianSampler(nh=4096, nz=64, hyper_config=hyper_config)
    decoder = HyperResNetDecoder(data_name, hyper_config)
    model = ImageModel(
        encoder=encoder, decoder=decoder, sampler=sampler, hyper_config=hyper_config)
    return model.to(device)


class HyperImageCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict):
        if "reconstruct_error" in outputs_dict:
            log_likelihood = -outputs_dict["reconstruct_error"]
        else:
            log_likelihood = -F.mse_loss(outputs_dict["inputs"], outputs_dict["logits"])
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
        if "beta" in outputs_dict:
            outputs_dict["beta"] = torch.ones(outputs_dict["beta"].shape).to(outputs_dict["beta"].device)
        elif "alpha" in outputs_dict:
            outputs_dict["alpha"] = 0.5 * torch.ones(outputs_dict["alpha"].shape).to(outputs_dict["alpha"].device)
        else:
            raise NotImplementedError
        return HyperImageCriterion.forward(outputs_dict)


def build_hyper_criterion(device):
    loss_fnc = HyperImageCriterion()
    return loss_fnc.to(device)
