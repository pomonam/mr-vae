import torch
from torch import nn
import torchvision


class HyperVae(nn.Module):

    def __init__(self, encoder, decoder, sampler):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def encode(self, x, beta, ignore_hyper=False):
        ix = self.encoder(x, beta, ignore_hyper)
        outputs_dict = self.sampler(ix, beta, ignore_hyper)
        return outputs_dict

    def decode(self, z, beta, ignore_hyper=False):
        return self.decoder(z, beta, ignore_hyper)

    def forward(self, x, beta, ignore_hyper=False):
        mu, std = self.encode(x, beta, ignore_hyper)
        z = self.sample(mu, std)
        rx = self.decode(z, beta, ignore_hyper)
        outputs_dict = {
            "inputs": x,
            "mean": mu,
            "stddev": std,
            "logits": rx
        }
        return outputs_dict

    def sample_inputs(self, batch):
        outputs_dict = self.model(batch["inputs"])
        recons = outputs_dict["logits"]
        torchvision.utils.save_image(recons)
