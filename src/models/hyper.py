from torch import nn
from src.models.blocks import MlpBlock, ResidualBlock, BatchNormResidualBlock, ConvNextBlock

_BLOCK_DICT = {
  "bn_residual": BatchNormResidualBlock,
  "residual": ResidualBlock,
  "mlp": MlpBlock,
  "convnext": ConvNextBlock
}


def replace_module(model: nn.Module, block_name) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module, block_name)

        if isinstance(module, nn.Linear):
            hyper_module = HyperModule(module, module.out_features, block_name)
            setattr(model, name, hyper_module)

        if isinstance(module, nn.Conv2d):
            # TODO: Change this
            hyper_module = HyperModule(module, block_name)
            setattr(model, name, hyper_module)


class HyperModule(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 # hyper_type: str,
                 width,
                 block_name: str):
        super().__init__()
        self.module = module
        self.hyper_type = "cond"
        self.hyper_module = _BLOCK_DICT[block_name](width)

    def forward(self, x, beta):
        x = self.module(x)

        if self.hyper_type == "cond":
            outputs_dict = self.hyper_module(beta)
            x = x * outputs_dict["scale"] + outputs_dict["bias"]
        else:
            x = x + self.hyper_module(beta)
        return x
