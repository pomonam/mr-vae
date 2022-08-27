from collections import OrderedDict

import torch
import torch.nn as nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder


class LstmEncoder(BaseEncoder):

    def __init__(self):
        BaseEncoder.__init__(self)
        self.latent_dim = 32

        self.embed = nn.Embedding(20001, 512)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=1,
            batch_first=True,
            dropout=0.)

        self.embedding = nn.Linear(1024, self.latent_dim)
        self.log_var = nn.Linear(1024, self.latent_dim)

        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x: torch.Tensor):
        output = OrderedDict()
        word_embed = self.embed(x)
        _, (last_state, last_cell) = self.lstm(word_embed)
        out = last_state.squeeze(0)
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
        return output


class LstmDecoder(BaseDecoder):

    def __init__(self):
        BaseDecoder.__init__(self)

        self.embed = nn.Embedding(20001, 512, padding_idx=-1)
        self.trans_linear = nn.Linear(32, 1024, bias=False)

        self.dropout_in = nn.Dropout(0.5)
        self.dropout_out = nn.Dropout(0.5)

        self.lstm = nn.LSTM(
            input_size=512 + 32,
            hidden_size=1024,
            num_layers=1,
            batch_first=True)

        self.pred_linear = nn.Linear(1024, 20001, bias=False)

        vocab_mask = torch.ones(20001)
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, z: torch.Tensor):
        raise LookupError

    def special_decode(self, input, z):
        batch_size, _ = z.size()
        seq_len = input.size(1)

        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        z_ = z.unsqueeze(1).expand(batch_size, seq_len, 32)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size, 32)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits

    def ar_forward(self, x, z):
        src = x[:, :-1]
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        output_logits = self.special_decode(src, z)
        tgt = tgt.contiguous().view(-1)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)), tgt)
        return loss.view(batch_size, -1).sum(-1)
