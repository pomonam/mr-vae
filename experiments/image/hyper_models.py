import torch
import torch.nn as nn
from src.models.base_encoder import BaseEncoder
from src.models.base_decoder import BaseDecoder
from src.hyper.layers.conv2d import HyperConv2d, HyperConvTranspose2d
from src.hyper.layers.linear import HyperLinear
from src.hyper.models import BaseHyperEncoder
from src.hyper.models import BaseHyperDecoder


def normal_init(m, mean, std):
  if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    m.weight.data.normal_(mean, std)
    if m.bias is not None:
      m.bias.data.zero_()


def weight_init(model, mean=0, std=0.02):
  for m in model._modules:
    normal_init(model._modules[m], mean, std)
  return


class HyperResNetEncoder(BaseHyperEncoder):
  def __init__(self, data_name, hyper_config):
    super().__init__()
    channels = 3
    bias = False
    c = 64

    self.data_name = data_name

    inc = c
    # self.ec0 = nn.Conv2d(channels, inc, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn0 = nn.BatchNorm2d(inc)
    self.ec0 = HyperConv2d(channels, inc, kernel_size=3, stride=1, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="leaky_relu")
    self.bn0 = nn.Identity()

    # self.ec1 = nn.Conv2d(inc, c, kernel_size=4, stride=2, padding=1, bias=bias)
    # self.bn1 = nn.BatchNorm2d(c)

    self.ec1 = HyperConv2d(inc, c, kernel_size=4, stride=2, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="none")
    self.bn1 = nn.Identity()

    # self.b11 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn11 = nn.BatchNorm2d(c)

    self.b11 = HyperConv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="leaky_relu")
    self.bn11 = nn.Identity()

    # self.b12 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn12 = nn.BatchNorm2d(c)
    self.b12 = HyperConv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="none")
    # self.bn12 = nn.BatchNorm2d(c)

    c = c * 2
    # self.ec2 = nn.Conv2d(c // 2, c, kernel_size=4, stride=2, padding=1, bias=bias)
    # self.bn2 = nn.BatchNorm2d(c)

    self.ec2 = HyperConv2d(c // 2, c, kernel_size=4, stride=2, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="none")

    # self.b21 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn21 = nn.BatchNorm2d(c)
    self.b21 = HyperConv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="leaky_relu")

    # self.b22 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    self.b22 = HyperConv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="none")
    # self.bn22 = nn.BatchNorm2d(c)

    c_out = c * 2
    if data_name == "celeba":
      self.ec3 = nn.Conv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=bias)
      self.bn3 = nn.BatchNorm2d(c_out)
      self.b31 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=bias)
      self.bn31 = nn.BatchNorm2d(c_out)
      self.b32 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=bias)
      self.bn32 = nn.BatchNorm2d(c_out)
      c = c_out

    # self.ec4 = nn.Conv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=False)
    self.ec4 = HyperConv2d(c, c_out, kernel_size=4, stride=2, padding=1, bias=False, apply_bn=True,
                           hyper_config=hyper_config, activation_fnc="none")
    self.act = nn.LeakyReLU(0.02)

  def forward(self, x):
    x = self.ec0(x)
    # x = self.bn0(x)
    # x = self.act(x)

    x = self.ec1(x)
    # x = self.bn1(x)
    y = x
    x = self.act(x)
    x = self.b11(x)
    # x = self.bn11(x)
    # x = self.act(x)
    x = self.b12(x)
    # x = self.bn12(x)
    x = self.act(x + y)

    x = self.ec2(x)
    # x = self.bn2(x)
    y = x
    x = self.act(x)
    x = self.b21(x)
    # x = self.bn21(x)
    # x = self.act(x)
    x = self.b22(x)
    # x = self.bn22(x)
    x = self.act(x + y)

    if self.data_name == "celeba":
      x = self.ec3(x)
      x = self.bn3(x)
      y = x
      x = self.act(x)
      x = self.b31(x)
      x = self.bn31(x)
      x = self.act(x)
      x = self.b32(x)
      x = self.bn32(x)
      x = self.act(x + y)

    x = self.ec4(x)
    x = x.view(x.size(0), -1)

    return x


class HyperResNetDecoder(BaseHyperDecoder):
  def __init__(self, data_name, hyper_config):
    super().__init__()
    channels = 3
    self.data_name = data_name
    self.hyper_config = hyper_config

    bias = False
    c = inch = 128

    self.in_channels = inch
    if self.data_name == "celeba":
      c = inch = 256
      self.dc1 = nn.ConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
      self.bn1 = nn.BatchNorm2d(c)
      self.b11 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
      self.bn11 = nn.BatchNorm2d(c)
      self.b12 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
      self.bn12 = nn.BatchNorm2d(c)
      inch = c
      c = c // 2

    # self.dc2 = nn.ConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
    # self.bn2 = nn.BatchNorm2d(c)
    self.dc2 = HyperConvTranspose2d(inch, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="none", hyper_config=hyper_config, apply_bn=True)
    # self.bn2 = nn.BatchNorm2d(c)

    # self.b21 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn21 = nn.BatchNorm2d(c)

    self.b21 = HyperConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="leaky_relu", hyper_config=hyper_config, apply_bn=True)

    # self.b22 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn22 = nn.BatchNorm2d(c)
    self.b22 = HyperConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="none", hyper_config=hyper_config, apply_bn=True)

    c = c // 2
    # self.dc3 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias)
    # self.bn3 = nn.BatchNorm2d(c)

    self.dc3 = HyperConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="none", hyper_config=hyper_config, apply_bn=True)

    # self.b31 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn31 = nn.BatchNorm2d(c)
    self.b31 = HyperConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias,
                                    activation_fnc="leaky_relu", hyper_config=hyper_config, apply_bn=True)

    # self.b32 = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, bias=bias)
    # self.bn32 = nn.BatchNorm2d(c)
    self.b32 = HyperConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="none", hyper_config=hyper_config, apply_bn=True)

    # self.dc4 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
    # self.bn4 = nn.BatchNorm2d(c)
    self.dc4 = HyperConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="leaky_relu", hyper_config=hyper_config, apply_bn=True)

    # self.dc4 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
    # self.bn4 = nn.BatchNorm2d(c)
    # self.dc4 = HyperConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1, output_padding=0, bias=bias,
    #                                 activation_fnc="leaky_relu", hyper_config=hyper_config, apply_bn=True)

    # self.dc5 = nn.ConvTranspose2d(c, channels, kernel_size=3, stride=1, padding=1, bias=False)
    self.dc5 = HyperConvTranspose2d(c, channels, kernel_size=3, stride=1, padding=1, output_padding=0, bias=bias,
                                    activation_fnc="none", hyper_config=hyper_config, apply_bn=False)

    self.act = nn.LeakyReLU(0.02)

    if self.data_name == "celeba":
      self.in_channels = self.dc1.in_channels
    else:
      self.in_channels = self.dc2.in_channels

    self.fmres = 4
    out_size = self.in_channels * self.fmres * self.fmres
    # bias = True
    # self.l1l = nn.Linear(64, out_size, bias=bias)
    self.l1l = HyperLinear(64, out_size, bias=True, hyper_config=hyper_config, activation_fnc="none")

  def forward(self, x):
    x = x.view(x.size(0), -1)

    x = self.l1l(x)
    x = x.view(x.size(0), self.in_channels, self.fmres, self.fmres)

    if self.data_name == "celeba":
      x = self.dc1(x)
      x = self.bn1(x)
      y = x
      x = self.act(x)
      x = self.b11(x)
      x = self.bn11(x)
      x = self.act(x)
      x = self.b12(x)
      x = self.bn12(x)
      x = self.act(x + y)

    x = self.dc2(x)
    # x = self.bn2(x)
    y = x
    x = self.act(x)
    x = self.b21(x)
    # x = self.bn21(x)
    # x = self.act(x)
    x = self.b22(x)
    # x = self.bn22(x)
    x = self.act(x + y)

    x = self.dc3(x)
    # x = self.bn3(x)
    y = x
    x = self.act(x)
    x = self.b31(x)
    # x = self.bn31(x)
    # x = self.act(x)
    x = self.b32(x)
    # x = self.bn32(x)
    x = self.act(x + y)

    x = self.dc4(x)
    # x = self.bn4(x)
    # x = self.act(x)
    x = self.dc5(x)

    return x
