from collections import OrderedDict

import torch
from torch import nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock


class Encoder(BaseEncoder):

  def __init__(self):
    BaseEncoder.__init__(self)

    self.fc1 = nn.Linear(4096, 1200)
    self.fc2 = nn.Linear(1200, 1200)
    self.fc3 = nn.Linear(1200, 10)
    self.fc4 = nn.Linear(1200, 10)

    self.act = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor):
    h = x.view(-1, 64 * 64)
    h = self.act(self.fc1(h))
    h = self.act(self.fc2(h))
    # z = h.view(x.size(0), 10)
    output = dict()
    output["embedding"] = self.fc3(h)
    output["log_covariance"] = self.fc4(h)
    return output


class Decoder(BaseDecoder):

  def __init__(self):
    BaseDecoder.__init__(self)

    self.net = nn.Sequential(
        nn.Linear(10, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 4096),
        nn.Sigmoid()
    )

  def forward(self, z):
      h = z.view(z.size(0), -1)
      h = self.net(h)
      mu_img = h.view(z.size(0), 1, 64, 64)
      output = dict()
      output["reconstruction"] = mu_img
      return output
