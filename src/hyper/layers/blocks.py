import math

import torch
from torch import nn
import torch.nn.functional as F


def get_block(name: str):
    _BLOCK_DICT = {
        "linear": LinearBlock,
        "mlp": MlpBlock,
        "residual": ResidualBlock,
        "bn_residual": BatchNormResidualBlock,
        "transformer": TransformerBlock
    }
    return _BLOCK_DICT[name]


class BaseBlock(nn.Module):

    def __init__(self, in_features: int, width: int):
        super().__init__()
        self.input_dim = in_features
        self.width = width

        self.layers = None
        self._construct_layers()

    def _construct_layers(self) -> None:
        raise NotImplementedError

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers = nn.Linear(self.input_dim, self.width)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.layers(beta)
        return out


class MlpBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.layers(beta)
        return out


class ResidualBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers1 = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        self.layers2 = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        self.temp_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=True), nn.ReLU())

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers1(out)
        out = out + self.layers2(out)
        return out


class BatchNormResidualBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
        )
        self.temp_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False), nn.ReLU())

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers(out)
        return out


class TransformerMlpBlock(nn.Module):

    def __init__(
            self,
            width: int,
            mlp_dim: int = None,  # Defaults to 4x input dim
            dropout: float = 0.0) -> None:
        super().__init__()

        self.width = width
        self.mlp_dim = mlp_dim or 4 * width
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(self.width, self.mlp_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_dim, self.width))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.normal_(std=1e-6)

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):
    """Self-attention special case of multi-head dot-product attention."""

    def __init__(self,
                 width: int,
                 num_heads: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.width = width
        self.num_heads = num_heads

        assert width % num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')

        self.head_dim = int(width / num_heads)
        self.all_head_dim = self.num_heads * self.head_dim

        self.query = nn.Linear(self.width, self.all_head_dim)
        torch.nn.init.xavier_uniform_(self.query.weight)
        self.query.bias.data.zero_()
        self.key = nn.Linear(self.width, self.all_head_dim)
        torch.nn.init.xavier_uniform_(self.key.weight)
        self.key.bias.data.zero_()
        self.value = nn.Linear(self.width, self.all_head_dim)
        torch.nn.init.xavier_uniform_(self.value.weight)
        self.value.bias.data.zero_()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.width, self.width)
        torch.nn.init.xavier_uniform_(self.out.weight)
        self.out.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        mixed_query_layer = self.query(x)

        key_layer = self.transpose_for_scores(self.key(x))
        value_layer = self.transpose_for_scores(self.value(x))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_dim,)
        context_layer = context_layer.view(new_context_layer_shape)
        out = self.out(context_layer)
        return out


class TransformerBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.project_feat = nn.Linear(self.input_dim, self.width)
        self.layer_norm0 = nn.LayerNorm(self.width)
        self.self_attention1 = SelfAttention(self.width, 12)
        self.layer_norm2 = nn.LayerNorm(self.width)
        self.mlp3 = TransformerMlpBlock(self.width, self.width * 4, 0.0)

    def forward(self, x):
        out = {}
        x = self.project_feat(x)
        y = self.layer_norm0(x)
        y = out['sa'] = self.self_attention1(y)
        # y = self.dropout(y)
        x = out['+sa'] = x + y

        y = self.layer_norm2(x)
        y = out['mlp'] = self.mlp3(y)
        # y = self.dropout(y)
        x = out['+mlp'] = x + y
        return x
