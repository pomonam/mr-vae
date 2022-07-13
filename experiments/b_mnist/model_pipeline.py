import torch
from torch import nn

from src.criterions import binary_cross_entropy
from src.criterions import kl_gaussian
from experiments.b_mnist.models.decoders import MLPDecoder, CNNDecoder, PixelCNNDecoder
from experiments.b_mnist.models.encoders import MLPEncoder, CNNEncoder, ResNetEncoder
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae


class BinarizedMnistMlpModel(BaseVae):
    def forward(self, x):
        outputs_dict = self.encode(x)
        z = self.sampler.sample(outputs_dict)
        logits = self.decode(z)
        outputs_dict = {
            "inputs": x,
            "mean": outputs_dict["mean"],
            "stddev": outputs_dict["stddev"],
            "logits": logits
        }
        return outputs_dict


# class HyperBinarizedMnistMlpModel(HyperVae):
#     def __init__(self, encoder, decoder, sampler, hyper_type, block_name):
#         super().__init__(encoder, decoder, sampler)
#
#         self.encoder = encoder
#         self.decoder = decoder
#         self.sampler = sampler
#         replace_module(self.encoder, hyper_type, block_name)
#         replace_module(self.decoder, hyper_type, block_name)
#         # replace_module(self.sampler, hyper_type, block_name)
#
#     def forward(self, x, beta, ignore_hyper=False):
#         outputs_dict = self.encode(x, beta, ignore_hyper)
#         z = self.sampler.sample(outputs_dict)
#         logits = self.decode(z, beta, ignore_hyper)
#         outputs_dict = {
#             "inputs": x,
#             "mean": outputs_dict["mean"],
#             "stddev": outputs_dict["stddev"],
#             "logits": logits
#         }
#         return outputs_dict


def build_model(encoder_name, decoder_name, device):
    if encoder_name == "mlp":
        encoder = MLPEncoder()
    elif encoder_name == "cnn":
        encoder = CNNEncoder()
    elif encoder_name == "resnet":
        encoder = ResNetEncoder()
    else:
        raise ValueError()

    sampler = IsotropicGaussianSampler(hidden_size=256, latent_size=64)

    if decoder_name == "mlp":
        decoder = MLPDecoder()
    elif decoder_name == "cnn":
        decoder = CNNDecoder()
    elif decoder_name == "pixelcnn":
        decoder = PixelCNNDecoder()
    else:
        raise ValueError()
    model = BinarizedMnistMlpModel(encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


# def build_hyper_model(hyper_type, block_name, device):
#     # The architecture is inspired from:
#     # https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py
#     encoder = MLPEncoder(structure=(784, 512))
#     sampler = HyperIsotropicGaussianSampler(hidden_size=512, latent_size=10,
#                                             hyper_type=hyper_type, block_name=block_name)
#     decoder = MLPDecoder(structure=(10, 512, 784))
#     model = HyperBinarizedMnistMlpModel(encoder=encoder, decoder=decoder, sampler=sampler,
#                                         hyper_type=hyper_type, block_name=block_name)
#     return model.to(device)


class BinarizedMnistMlpCriterion(nn.Module):
    def get_metric_lst(self):
        return ["loss", "rate", "distortion"]

    def forward(self, outputs_dict, beta):
        log_likelihood = -binary_cross_entropy(outputs_dict["inputs"], outputs_dict["logits"])
        kl = kl_gaussian(outputs_dict["mean"], torch.square(outputs_dict["stddev"]))
        if isinstance(beta, int) or isinstance(beta, float):
            elbo = log_likelihood - beta * kl
        else:
            elbo = log_likelihood - beta.view(-1) * kl

        loss_dict = {
            "loss": -torch.mean(elbo).item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return -torch.mean(elbo), loss_dict


def build_criterion(device):
    loss_fnc = BinarizedMnistMlpCriterion()
    return loss_fnc.to(device)
