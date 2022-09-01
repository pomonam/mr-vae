import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.blocks import get_block


# class HyperLayer(nn.Module):
#
#   def __init__(self, features: int, hyper_cfg: HyperConfig, use_group=True):
#     super().__init__()
#
#     self._net_inputs = None
#     self.features = features
#     self.hyper_cfg = hyper_cfg
#
#     # input_dim = hyper_cfg.preprocess_dim if hyper_cfg.preprocess_beta else 1
#     # self.hyper_block_scale = get_block("linear")(input_dim, self.features)
#     # self.hyper_block_shift = get_block("linear")(input_dim, self.features)
#
#     if hyper_cfg.preprocess_beta:
#       self.hyper_block_scale = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
#       self.hyper_block_shift = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
#     else:
#       self.hyper_block_scale = get_block(self.hyper_cfg.block_type)(
#         in_features=1, width=self.hyper_cfg.preprocess_dim)
#       self.hyper_block_shift = get_block(self.hyper_cfg.block_type)(
#         in_features=1, width=self.hyper_cfg.preprocess_dim)
#
#     if self.hyper_cfg.include_layer_norm:
#       if use_group:
#         self.layer_norm = torch.nn.GroupNorm(1, self.features, affine=False)
#       else:
#         self.layer_norm = torch.nn.LayerNorm(
#             self.features, elementwise_affine=False)
#     else:
#       self.layer_norm = nn.Identity()
#
#   def set_net_inputs(self, value: torch.Tensor) -> None:
#     self._net_inputs = value
#
#   def reset_net_inputs(self) -> None:
#     self._net_inputs = None
#
#   def forward(self, inputs):
#     scale = self.hyper_block_scale(self._net_inputs)
#     if self.hyper_cfg.include_sigmoid_activation:
#       scale = torch.sigmoid(scale)
#     shift = self.hyper_block_shift(self._net_inputs)
#
#     if len(inputs.shape) == 4:
#       scale = scale.unsqueeze(-1).unsqueeze(-1)
#       shift = shift.unsqueeze(-1).unsqueeze(-1)
#
#     out = scale * self.layer_norm(inputs)
#     if self.hyper_cfg.include_shift:
#       out = out + shift
#
#     if self.hyper_cfg.include_residual_connection:
#       return inputs + out
#     else:
#       return out
