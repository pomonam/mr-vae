from torch import nn


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

    def decode(self, z):
        return self.decoder(z)

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


