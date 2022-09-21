import torch
import torch.nn as nn
import math


class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(LSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size

    self.weight = nn.Parameter(torch.empty(4 * hidden_size, input_size + hidden_size + 2))

    stdv = 1.0 / math.sqrt(self.hidden_size)
    torch.nn.init.uniform_(self.weight, -stdv, stdv)

  def forward(self, input, state):
    hidden_list = []

    hx, cx = state

    inputs = input.unbind(0)
    for i in range(len(inputs)):
      hx = hx.squeeze()
      cx = cx.squeeze()

      weight_ih = self.weight[:, :self.input_size]
      weight_hh = self.weight[:, self.input_size:self.hidden_size + self.input_size]
      bias_ih = self.weight[:, -2]
      bias_hh = self.weight[:, -1]

      cur_weight_hh = torch.nn.functional.dropout(weight_hh, p=0.5, training=self.training)

      # else:
      #     m = weight_hh.data.new(weight_hh.size()).fill_(
      #         1)  # All 1's (nothing dropped) at test-time
      #     mask = Variable(m, requires_grad=False)
      # weight_hh = weight_hh * mask

      x = inputs[i]
      gates = torch.matmul(x, weight_ih.t()) + bias_ih + torch.matmul(hx, cur_weight_hh.t()) + bias_hh
      ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

      ingate = torch.sigmoid(ingate)
      forgetgate = torch.sigmoid(forgetgate)
      cellgate = torch.tanh(cellgate)
      outgate = torch.sigmoid(outgate)

      cx = (forgetgate * cx) + (ingate * cellgate)
      hx = outgate * torch.tanh(cx)
      hidden_list.append(hx)

    return torch.stack(hidden_list), (hx.unsqueeze(0), cx.unsqueeze(0))


class LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, tie_weights=False):
        super(LSTM, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else
        (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = self.encoder(input)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                outputs.append(raw_output)
        hidden = new_hidden

        output = raw_output
        outputs.append(raw_output)

        result = output.view(output.size(0 ) *output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                for l in range(self.nlayers)]

