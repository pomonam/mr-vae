from collections import OrderedDict

import torch
from torch import nn
import texar.torch as tx

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock


class TransformerDecoder(BaseDecoder):

  def __init__(self):
    super().__init__()

    vocab_size = 2
    seq_length = 28 * 28
    self.hidden_size = 512
    dec_emb_hparams = {
      'name': 'lookup_table',
      "dim": self.hidden_size,
      "dropout_rate": 0.,
      'initializer': {
        'type': 'normal_',
        'kwargs': {
          'mean': 0.0,
          'std': self.hidden_size ** -0.5,
        },
      }
    }
    self.decoder_w_embedder = tx.modules.WordEmbedder(
      vocab_size=vocab_size, hparams=dec_emb_hparams)

    dec_pos_emb_hparams = {
      'dim': self.hidden_size,
    }
    self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
      position_size=seq_length,
      hparams=dec_pos_emb_hparams)

    relu_dropout = 0.2
    embedding_dropout = 0.2
    attention_dropout = 0.2
    residual_dropout = 0.2
    num_blocks = 3
    trans_hparams = {
      'output_layer_bias': False,
      'embedding_dropout': embedding_dropout,
      'residual_dropout': residual_dropout,
      'num_blocks': num_blocks,
      'dim': self.hidden_size,
      'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
          'factor': 1.0,
          'mode': 'FAN_AVG',
          'uniform': True,
        },
      },
      'multihead_attention': {
        'dropout_rate': attention_dropout,
        'num_heads': 8,
        'num_units': self.hidden_size,
        'output_dim': self.hidden_size
      },
      'poswise_feedforward': {
        'name': 'fnn',
        'layers': [
          {
            'type': 'Linear',
            'kwargs': {
              "in_features": self.hidden_size,
              "out_features": self.hidden_size * 4,
              "bias": True,
            },
          },
          {
            'type': 'ReLU',
          },
          {
            'type': 'Dropout',
            'kwargs': {
              'p': relu_dropout,
            }
          },
          {
            'type': 'Linear',
            'kwargs': {
              "in_features": self.hidden_size * 4,
              "out_features": self.hidden_size,
              "bias": True,
            }
          }
        ],
      }
    }

    self.transformer_decoder = tx.modules.TransformerDecoder(
      # tie word embedding with output layer
      output_layer=self.decoder_w_embedder.embedding,
      token_pos_embedder=self._embed_fn_transformer,
      hparams=trans_hparams)

    self.mlp_linear_layer = nn.Linear(16, self.hidden_size, bias=True)
    self.output_layer = nn.Linear(2, 1, bias=True)

  def _embed_fn_transformer(self,
                            tokens: torch.LongTensor,
                            positions: torch.LongTensor):
    r"""Generates word embeddings combined with positional embeddings
      """
    output_p_embed = self.decoder_p_embedder(positions)
    output_w_embed = self.decoder_w_embedder(tokens.long())
    output_w_embed = output_w_embed * self.hidden_size ** 0.5
    output_embed = output_w_embed + output_p_embed
    return output_embed

  def forward(self, z: torch.Tensor):
    raise LookupError

  def decode(self,
             helper,
             latent_z,
             text_ids,
             seq_lengths,
             max_decoding_length=None):
    self._latent_z = latent_z
    fc_output = self.mlp_linear_layer(latent_z)

    transformer_states = fc_output.unsqueeze(1)
    outputs = self.transformer_decoder(
      inputs=text_ids,
      memory=transformer_states,
      memory_sequence_length=torch.ones(transformer_states.size(0)),
      helper=helper,
      max_decoding_length=max_decoding_length)
    return outputs

  def ar_forward(self, x, z, batch):
    data_batch = x
    self._latent_z = z

    x_flatten = x.reshape(-1, 784)
    x_flatten = torch.cat((torch.zeros((x_flatten.shape[0], 1)), x_flatten[:, :-1]), 1)
    seq_lengths = 784 - 1
    outputs = self.decode(
      helper=None, latent_z=z,
      text_ids=x_flatten[:, :-1], seq_lengths=seq_lengths)

    logits = outputs.logits
    logits = self.output_layer(logits)
    logits = torch.sigmoid(logits)

    rc_loss = torch.nn.functional.mse_loss(logits.squeeze(-1),
                                           x_flatten[:, 1:],
                                           reduction="none",
                                           ).sum(dim=-1)

    return rc_loss
