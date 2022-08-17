from src.models.base_encoder import BaseEncoder
import torch
from torch import nn

from src.models.base_decoder import BaseDecoder
from experiments.text.models import uniform_initializer, xavier_normal_initializer
from src.hyper.layers.scale import HyperScale
from src.hyper.layers.linear import HyperLinear
from src.hyper.models import BaseHyperEncoder
from src.hyper.models import BaseHyperDecoder


model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)


class HyperLstmEncoder(BaseHyperEncoder):

    def __init__(self, vocab_size, ni, nh, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config
        self.embed = nn.Embedding(vocab_size, ni)
        # self.scale = HyperScale(ni, "none", hyper_config=hyper_config)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            batch_first=True)

        # self.linear = nn.Linear(nh, nh, bias=True)
        self.linear = HyperLinear(nh, nh, "none", hyper_config)
        self.linear_proj = nn.Linear(nh, nh, bias=True)

        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, x):
        word_embed = self.embed(x)
        # word_embed = self.scale(word_embed)
        _, (last_state, last_cell) = self.lstm(word_embed)
        out = self.linear(last_state.squeeze(0))
        out = self.linear_proj(out)
        return out


class HyperLstmDecoder(BaseHyperDecoder):

    def __init__(self, vocab_size, ni, nh, nz, hyper_config):
        super().__init__()

        self.vocab_size = vocab_size
        self.ni = ni
        self.nh = nh
        self.nz = nz

        self.embed = nn.Embedding(vocab_size, ni, padding_idx=-1)
        # self.scale = HyperScale(ni, "none", hyper_config=hyper_config)
        self.trans_linear = HyperLinear(nz, nh, "none", hyper_config=hyper_config, bias=False)

        self.lstm = nn.LSTM(input_size=ni + nz,
                            hidden_size=nh,
                            num_layers=1,
                            batch_first=True)

        self.pred_linear = HyperLinear(nh, vocab_size, "none", hyper_config=hyper_config, bias=False)
        # self.pred_linear = nn.Linear(nh, vocab_size, bias=False)

        vocab_mask = torch.ones(vocab_size)
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, nz)
        """

        # not predicting start symbol
        # sents_len -= 1

        batch_size, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        # word_embed = self.scale(word_embed)
        # word_embed = self.dropout_in(word_embed)

        z_ = z.unsqueeze(1).expand(batch_size, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        # h_init = self.trans_linear(z).unsqueeze(0)
        # c_init = h_init.new_zeros(h_init.size())
        output, _ = self.lstm(word_embed, (h_init, c_init))

        # output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        # self.pred_linear._net_inputs = self.pred_linear._net_inputs.repeat(1, output.shape[1]).view(-1).unsqueeze(-1)
        self.pred_linear._net_inputs = self.pred_linear._net_inputs.unsqueeze(1).repeat(1, output.shape[1], 1).reshape(-1, 784)
        output_logits = self.pred_linear(output.reshape(-1, output.shape[2])).reshape(output.shape[0], output.shape[1], -1)
        # output_logits = self.pred_linear(output).reshape(output.shape[0], output.shape[1], -1)
        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """
        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        # n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        tgt = tgt.contiguous().view(-1)

        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)), tgt)

        return loss.view(batch_size, -1).sum(-1)
