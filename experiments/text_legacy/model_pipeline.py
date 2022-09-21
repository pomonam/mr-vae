import torch
from torch import nn
import math
from experiments.text_legacy.models import LstmDecoder, LstmEncoder
from experiments.text_legacy.old_version.hyper.vae import HyperVae
from experiments.text_legacy.old_version.models.samplers import IsotropicGaussianSampler
from experiments.text_legacy.old_version.models.vae import BaseVae
from experiments.text_legacy.old_version.criterions import kl_gaussian, log_sum_exp
from torch.nn import functional as F

from experiments.text_legacy.hyper_models import HyperLstmDecoder, HyperLstmEncoder
from experiments.text_legacy.old_version.hyper.models import HyperIsotropicGaussianSampler


class TextModel(BaseVae):

    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        reconstruct_err = self.decoder.reconstruct_error(x, z)
        outputs_dict = {
            "inputs": x,
            "mean": outputs_dict["mean"],
            "log_var": outputs_dict["log_var"],
            "reconstruct_err": reconstruct_err
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


class HyperTextModel(HyperVae):

    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        reconstruct_err = self.decoder.reconstruct_error(x, z)
        outputs_dict = {
            "inputs": x,
            "mean": outputs_dict["mean"],
            "log_var": outputs_dict["log_var"],
            "reconstruct_err": reconstruct_err
        }
        return outputs_dict


def build_model(name, device):
    if name == "yahoo":
        vocab_size = 20001
    elif name == "yelp":
        vocab_size = 19997
    else:
        raise Exception

    encoder = LstmEncoder(vocab_size, 512, 1024)
    sampler = IsotropicGaussianSampler(nh=1024, nz=32)
    decoder = LstmDecoder(vocab_size, 512, 1024, 32)
    model = TextModel(encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


def build_hyper_model(name, hyper_config, device):
    if name == "yahoo":
        vocab_size = 20001
    elif name == "yelp":
        vocab_size = 19997
    else:
        raise Exception

    encoder = HyperLstmEncoder(vocab_size, 512, 1024, hyper_config)
    sampler = HyperIsotropicGaussianSampler(nh=1024, nz=32, hyper_config=hyper_config)
    decoder = HyperLstmDecoder(vocab_size, 512, 1024, 32, hyper_config=hyper_config)
    model = HyperTextModel(
        encoder=encoder, decoder=decoder, sampler=sampler, hyper_config=hyper_config)
    return model.to(device)


class TextCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict, beta: float = 1.0):
        log_likelihood = -outputs_dict["reconstruct_err"]
        kl = kl_gaussian(outputs_dict["mean"], outputs_dict["log_var"].exp())
        elbo = log_likelihood - beta * kl
        elbo = -torch.mean(elbo)
        loss_dict = {
            "loss": elbo,
            "distortion": -torch.mean(log_likelihood),
            "rate": torch.mean(kl)
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return TextCriterion.forward(outputs_dict, beta=1.0)


class HyperTextCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(outputs_dict: dict):
        log_likelihood = -outputs_dict["reconstruct_err"]
        kl = kl_gaussian(outputs_dict["mean"], outputs_dict["log_var"].exp())

        if "beta" in outputs_dict:
            elbo = log_likelihood - outputs_dict["beta"].view(-1) * kl
        elif "alpha" in outputs_dict:
            alpha = outputs_dict["alpha"].view(-1)
            elbo = alpha * log_likelihood - (1 - alpha) * kl
        else:
            raise NotImplementedError

        # elbo = log_likelihood - beta * kl
        elbo = -torch.mean(elbo)
        loss_dict = {
            "loss": elbo,
            "distortion": -torch.mean(log_likelihood),
            "rate": torch.mean(kl)
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return TextCriterion.forward(outputs_dict)


def build_criterion(device):
    loss_fnc = TextCriterion()
    return loss_fnc.to(device)


def build_hyper_criterion(device):
    loss_fnc = HyperTextCriterion()
    return loss_fnc.to(device)