import torch
from torch import nn


class MlpBlock(nn.Module):

    def __init__(self, width: int, rank: int = None):
        super().__init__()
        self.width = width
        self.mid_width = width * 2 if rank is None else rank

        self.net = nn.Sequential(
            nn.Linear(1, self.mid_width),
            nn.GELU(),
            nn.Linear(self.mid_width, self.mid_width),
            nn.GELU(),
            nn.Linear(self.mid_width, self.mid_width)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.normal_(std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        scale, bias = out[:, :self.width], out[:, self.width:]
        outputs_dict = {
            "scale": scale,
            "bias": bias
        }
        return outputs_dict


class ResidualBlock(nn.Module):

    def __init__(self, width: int, rank: int = None):
        super().__init__()
        self.width = width
        self.mid_width = width if rank is None else rank

        self.linear1 = nn.Linear(1, self.mid_width, bias=False)
        self.linear2 = nn.Linear(self.mid_width, self.mid_width, bias=False)
        self.linear3 = nn.Linear(self.mid_width, self.width, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.linear3.weight, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(inputs)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)

        x = self.linear3(x)
        return x + inputs


class BatchNormResidualBlock(nn.Module):

    def __init__(self, width: int, rank: int = None):
        super().__init__()
        self.width = width
        self.mid_width = width if rank is None else rank

        self.linear1 = nn.Linear(self.width, self.mid_width, bias=False)
        self.bn1 = nn.BatchNorm1d(self.mid_width)
        self.linear2 = nn.Linear(self.mid_width, self.mid_width, bias=False)
        self.bn2 = nn.BatchNorm1d(self.mid_width)
        self.linear3 = nn.Linear(self.mid_width, self.width, bias=False)
        self.bn3 = nn.BatchNorm1d(self.width)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.bn3(x)
        return x + inputs


class ConvNextBlock(nn.Module):

    def __init__(self, width: int, rank: int = None):
        super().__init__()
        self.width = width
        self.mid_width = width * 4 if rank is None else rank

        self.linear1 = nn.Linear(1, self.width, bias=False)
        self.ln = nn.LayerNorm(self.width)
        self.linear2 = nn.Linear(self.width, self.mid_width, bias=False)
        self.linear3 = nn.Linear(self.mid_width, self.width, bias=False)
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.linear3.weight, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear1(inputs)
        x = self.ln(x)

        x = self.linear2(x)
        x = self.gelu(x)

        x = self.linear3(x)
        return x + inputs
