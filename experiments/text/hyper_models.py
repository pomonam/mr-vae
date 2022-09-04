from collections import OrderedDict
from src.hyper.layers import get_hyper_layer
import torch
import torch.nn as nn
import texar.torch as tx

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from experiments.text.hyper_modules.transformer import TransformerDecoder


class HyperLstmEncoder(BaseHyperEncoder):

  def __init__(self, vocab_size, hyper_cfg, v1=True):
    BaseHyperEncoder.__init__(self)

    self.latent_dim = 32
    self.hyper_cfg = hyper_cfg

    embed_dim = 512 if v1 else 256
    enc_emb_hparams = {
      'name': 'lookup_table',
      "dim": embed_dim,
      "dropout_rate": 0.,
      'initializer': {
        'type': 'normal_',
        'kwargs': {
          'mean': 0.0,
          'std': embed_dim ** -0.5,
        },
      }
    }
    self.embed = tx.modules.WordEmbedder(
      vocab_size=vocab_size, hparams=enc_emb_hparams)
    self.hyper_embed = get_hyper_layer(embed_dim, hyper_cfg)

    hidden_size = 550 if v1 else 256
    enc_cell_hparams = {
      "type": "LSTMCell",
      "kwargs": {
        "num_units": hidden_size,
        "bias": 0.
      },
      "num_layers": 1
    }
    self.encoder = tx.modules.UnidirectionalRNNEncoder[tx.core.LSTMState](
      input_size=self.embed.dim,
      hparams={
        "rnn_cell": enc_cell_hparams,
      })
    self.hyper_encoder = get_hyper_layer(hidden_size * 2, hyper_cfg)

    self.embedding = nn.Linear(hidden_size * 2, self.latent_dim)
    self.log_var = nn.Linear(hidden_size * 2, self.latent_dim)

  def forward(self, batch):
    text_ids = batch["text_ids"]

    output = OrderedDict()
    input_embed = self.embed(text_ids)
    input_embed = self.hyper_embed(input_embed)
    _, encoder_states = self.encoder(
      input_embed,
      sequence_length=batch["length"])
    out = torch.cat(encoder_states, 1)
    out = self.hyper_encoder(out)
    output["embedding"] = self.embedding(out)
    output["log_covariance"] = self.log_var(out)
    return output


class HyperLstmDecoder(BaseHyperDecoder):

  def __init__(self, vocab_size, hyper_cfg, v1=True):
    BaseHyperDecoder.__init__(self)
    self.hyper_cfg = hyper_cfg

    embed_dim = 512 if v1 else 256
    dec_emb_hparams = {
      'name': 'lookup_table',
      "dim": embed_dim,
      "dropout_rate": 0.5,
      'initializer': {
        'type': 'normal_',
        'kwargs': {
          'mean': 0.0,
          'std': embed_dim ** -0.5,
        },
      }
    }
    self.decoder_w_embedder = tx.modules.WordEmbedder(
      vocab_size=vocab_size, hparams=dec_emb_hparams)
    self.hyper_decoder_w_embedder = get_hyper_layer(embed_dim, hyper_cfg)

    hidden_size = 550 if v1 else 256
    self.hidden_size = hidden_size
    dec_cell_hparams = {
      "type": "LSTMCell",
      "kwargs": {
        "num_units": hidden_size,
        "bias": 0.
      },
      "dropout": {"output_keep_prob": 1. - 0.5},
      "num_layers": 1
    }
    self.lstm_decoder = tx.modules.BasicRNNDecoder(
      input_size=(self.decoder_w_embedder.dim + 32),
      vocab_size=vocab_size,
      token_embedder=self._embed_fn_rnn,
      hparams={"rnn_cell": dec_cell_hparams})
    # self.hyper_lstm_decoder = get_hyper_layer(hidden_size, hyper_cfg)

    self.mlp_linear_layer = nn.Linear(32, hidden_size * 2)
    self.hyper_mlp_linear_layer = get_hyper_layer(hidden_size * 2, hyper_cfg)

  def _embed_fn_rnn(self, tokens: torch.LongTensor):
    embedding = self.decoder_w_embedder(tokens)
    embedding = self.hyper_decoder_w_embedder(embedding)
    latent_z = self._latent_z
    if len(embedding.size()) > 2:
      latent_z = latent_z.unsqueeze(0).repeat(tokens.size(0), 1, 1)
    return torch.cat([embedding, latent_z], dim=-1)

  def decode(self,
             helper,
             latent_z,
             text_ids,
             seq_lengths,
             max_decoding_length=None):
    fc_output = self.mlp_linear_layer(latent_z)
    fc_output = self.hyper_mlp_linear_layer(fc_output)

    lstm_states = torch.chunk(fc_output, 2, dim=1)
    outputs, _, _ = self.lstm_decoder(
      initial_state=lstm_states,
      inputs=text_ids,
      helper=helper,
      sequence_length=seq_lengths,
      max_decoding_length=max_decoding_length)

    return outputs

  def ar_forward(self, x, z, batch):
    data_batch = x
    self._latent_z = z

    helper = self.lstm_decoder.create_helper(
      decoding_strategy="train_greedy",
      start_tokens=batch["start_tokens"],
      end_token=batch["end_token"]
    )

    seq_lengths = data_batch["length"] - 1
    outputs = self.decode(
      helper=helper, latent_z=z,
      text_ids=data_batch["text_ids"][:, :-1], seq_lengths=seq_lengths)

    logits = outputs.logits

    # Losses & train ops
    rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
      labels=data_batch["text_ids"][:, 1:], logits=logits,
      sequence_length=seq_lengths)

    return rc_loss


class HyperTransformerDecoder(BaseHyperDecoder):

  def __init__(self, vocab_size, hyper_cfg, v1=True):
    BaseHyperDecoder.__init__(self)

    embd_dim = 512 if v1 else 256
    hidden_size = 512 if v1 else 256
    self.hidden_size = hidden_size
    dec_emb_hparams = {
      'name': 'lookup_table',
      "dim": embd_dim,
      "dropout_rate": 0.,
      'initializer': {
        'type': 'normal_',
        'kwargs': {
          'mean': 0.0,
          'std': embd_dim ** -0.5,
        },
      }
    }
    self.decoder_w_embedder = tx.modules.WordEmbedder(
      vocab_size=vocab_size, hparams=dec_emb_hparams)
    self.hyper_decoder_w_embedder = get_hyper_layer(embd_dim, hyper_cfg)

    dec_pos_emb_hparams = {
      'dim': hidden_size,
    }
    self.decoder_p_embedder = tx.modules.SinusoidsPositionEmbedder(
      position_size=300,
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
      'dim': hidden_size,
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
        'num_units': hidden_size,
        'output_dim': hidden_size
      },
      'poswise_feedforward': {
        'name': 'fnn',
        'layers': [
          {
            'type': 'Linear',
            'kwargs': {
              "in_features": hidden_size,
              "out_features": hidden_size * 4,
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
              "in_features": hidden_size * 4,
              "out_features": hidden_size,
              "bias": True,
            }
          }
        ],
      }
    }

    # self.transformer_decoder = tx.modules.TransformerDecoder(
    #   # tie word embedding with output layer
    #   output_layer=self.decoder_w_embedder.embedding,
    #   token_pos_embedder=self._embed_fn_transformer,
    #   hparams=trans_hparams)
    self.transformer_decoder = TransformerDecoder(
      hyper_cfg=hyper_cfg,
      # tie word embedding with output layer
      output_layer=self.decoder_w_embedder.embedding,
      token_pos_embedder=self._embed_fn_transformer,
      hparams=trans_hparams)

    self.mlp_linear_layer = nn.Linear(32, hidden_size, bias=True)
    self.hyper_mlp_linear_layer = get_hyper_layer(hidden_size, hyper_cfg)

  def _embed_fn_transformer(self,
                            tokens: torch.LongTensor,
                            positions: torch.LongTensor):
    output_p_embed = self.decoder_p_embedder(positions)
    output_w_embed = self.decoder_w_embedder(tokens)
    output_w_embed = self.hyper_decoder_w_embedder(output_w_embed)
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
    fc_output = self.hyper_mlp_linear_layer(fc_output)

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

    seq_lengths = data_batch["length"] - 1
    outputs = self.decode(
      helper=None, latent_z=z,
      text_ids=data_batch["text_ids"][:, :-1], seq_lengths=seq_lengths)

    logits = outputs.logits

    # Losses & train ops
    rc_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
      labels=data_batch["text_ids"][:, 1:], logits=logits,
      sequence_length=seq_lengths)

    return rc_loss
