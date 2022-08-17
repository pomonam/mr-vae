import torch
from torch import nn

from experiments.text.models import LstmDecoder, LstmEncoder
from src.hyper.vae import HyperVae
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae
from src.criterions import kl_gaussian
from torch.nn import functional as F


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
    model = TextModel(
        encoder=encoder, decoder=decoder, sampler=sampler)
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
            "loss": elbo.item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return elbo, loss_dict

    @staticmethod
    def eval_forward(outputs_dict: dict):
        return TextCriterion.forward(outputs_dict, beta=1.0)


def build_criterion(device):
    loss_fnc = TextCriterion()
    return loss_fnc.to(device)
