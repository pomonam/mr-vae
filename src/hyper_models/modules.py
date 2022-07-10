from torch import nn
from src.hyper_models.blocks import LinearBlock, MlpBlock, ResidualBlock, BatchNormResidualBlock
import torch

_BLOCK_DICT = {
    "linear": LinearBlock,
    "bn_residual": BatchNormResidualBlock,
    "residual": ResidualBlock,
    "mlp": MlpBlock,
}


def replace_module(model: nn.Module, hyper_type, block_name) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module, hyper_type, block_name)

        if isinstance(module, nn.Linear):
            hyper_module = HyperModule(module, hyper_type, block_name)
            setattr(model, name, hyper_module)

        if isinstance(module, nn.Conv2d):
            # TODO: Change this
            hyper_module = HyperModule(module, block_name)
            setattr(model, name, hyper_module)


class HyperModule(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 hyper_type: str = "add",
                 block_name: str = "linear"):
        super().__init__()

        if hyper_type not in ["add", "mult"]:
            raise ValueError("Invalid hyper_type {}".format(str(hyper_type)))
        self.hyper_type = hyper_type
        self.block_name = block_name
        if isinstance(module, nn.Linear):
            self.module = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            self.hyper_module = _BLOCK_DICT[block_name](module.out_features)

        if self.hyper_type == "add":
            self.add_hyper_module = nn.Linear(module.in_features, module.out_features, bias=False)
        else:
            self.add_hyper_module = None

    def forward(self, inputs: torch.Tensor, beta: torch.Tensor, ignore_hyper: bool = False) -> torch.Tensor:
        if self.hyper_type == "add":
            if ignore_hyper:
                out = self.module(inputs)
            else:
                add_out = self.add_hyper_module(inputs)
                hyper_out = self.hyper_module(beta)
                hyper_out = add_out * hyper_out
                out = self.module(inputs) + hyper_out

        elif self.hyper_type == "mult":
            if ignore_hyper:
                print("Warning: hyper_type mult does not support ignore_hyper")
            hyper_out = self.hyper_module(beta)
            out = self.module(inputs) * hyper_out

        else:
            raise NotImplementedError

        return out
