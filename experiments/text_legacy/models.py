from src.models.base_encoder import BaseEncoder
import torch
from torch import nn

from src.models.base_decoder import BaseDecoder


class UniformInitializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


model_init = UniformInitializer(0.01)
emb_init = UniformInitializer(0.1)


class LstmEncoder(BaseEncoder):

    def __init__(self, vocab_size, ni, nh):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, ni)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.)

        # for param in self.parameters():
        #     model_init(param)
        # emb_init(self.embed.weight)

    def forward(self, x):
        word_embed = self.embed(x)
        _, (last_state, last_cell) = self.lstm(word_embed)
        return last_state.squeeze(0)


class LstmDecoder(BaseDecoder):

    def __init__(self, vocab_size, ni, nh, nz):
        super().__init__()

        self.vocab_size = vocab_size
        self.ni = ni
        self.nh = nh
        self.nz = nz

        self.embed = nn.Embedding(vocab_size, ni, padding_idx=-1)
        self.trans_linear = nn.Linear(nz, nh, bias=False)

        self.dropout_in = nn.Dropout(0.5)
        self.dropout_out = nn.Dropout(0.5)

        self.lstm = nn.LSTM(input_size=ni + nz,
                            hidden_size=nh,
                            num_layers=1,
                            batch_first=True)

        self.pred_linear = nn.Linear(nh, vocab_size, bias=False)

        vocab_mask = torch.ones(vocab_size)
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)

        # for param in self.parameters():
        #     model_init(param)
        # emb_init(self.embed.weight)

    def special_decode(self, input, z):
        batch_size, _ = z.size()
        seq_len = input.size(1)

        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        z_ = z.unsqueeze(1).expand(batch_size, seq_len, self.nz)
        word_embed = torch.cat((word_embed, z_), -1)

        z = z.view(batch_size, self.nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z, *argv):
        src = x[:, :-1]
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        output_logits = self.special_decode(src, z)
        tgt = tgt.contiguous().view(-1)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)), tgt)
        return loss.view(batch_size, -1).sum(-1)
