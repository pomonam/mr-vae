from torch import nn
from src.models.blocks import MlpBlock, ResidualBlock, BatchNormResidualBlock, ConvNextBlock
from src.utils import set_attr, del_attr
import torch

_BLOCK_DICT = {
  "bn_residual": BatchNormResidualBlock,
  "residual": ResidualBlock,
  "mlp": MlpBlock,
  "convnext": ConvNextBlock
}


# def replace_module(module, block_name):
#     for attr_str in dir(module):
#         target_attr = getattr(module, attr_str)
#         if type(target_attr) == torch.nn.Linear:
#             # print('replaced: ', name, attr_str)
#             new_bn = HyperModule(module, module.out_features, block_name)
#             setattr(module, attr_str, new_bn)
#
#     for name, immediate_child_module in module.named_children():
#         replace_module(immediate_child_module, name)


def replace_module(model: nn.Module, block_name) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module, block_name)

        if isinstance(module, nn.Linear):
            # del_attr(model, name)
            hyper_module = HyperModule(module, module.out_features, block_name)
            setattr(model, name, hyper_module)

        if isinstance(module, nn.Conv2d):
            hyper_module = HyperModule(module, block_name)
            setattr(model, name, hyper_module)


class HyperModule(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 width,
                 block_name: str):
        super().__init__()
        self.module = module
        self.hyper_module = _BLOCK_DICT[block_name](width)

    def forward(self, x, beta):
        x = self.module(x)
        x = x + self.hyper_module(beta)
        return x
