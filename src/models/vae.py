from torch import nn

from src.models.base_decoder import BaseDecoder
from src.models.base_encoder import BaseEncoder
from src.models.samplers import BaseSampler


class BaseVae(nn.Module):
    def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder,
                 sampler: BaseSampler):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def encode(self, x):
        ix = self.encoder(x)
        outputs_dict = self.sampler(ix)
        return outputs_dict

    def decode(self, z, **kwargs):
        return self.decoder(z, **kwargs)

    def forward(self, x):
        raise NotImplementedError
