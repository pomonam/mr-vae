import torch
from torch import nn
import torchvision


class BaseVae(nn.Module):

    def __init__(self, encoder, decoder, sampler):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def encode(self, x):
        ix = self.encoder(x)
        outputs_dict = self.sampler(ix)
        return outputs_dict

    def decode(self, x, z):
        return self.decoder(x, z)

    def forward(self, x):
        raise NotImplementedError
        # outputs_dict = self.encode(x)
        # z = self.sampler.sample(outputs_dict)
        # rx = self.decode(z)
        # outputs_dict = {
        #     "inputs": x,
        #     "mean": mu,
        #     "stddev": std,
        #     "logits": rx
        # }
        # return outputs_dict

    # def sample_inputs(self, batch):
    #     outputs_dict = self.model(batch["inputs"])
    #     recons = outputs_dict["logits"]
    #     torchvision.utils.save_image(recons)
