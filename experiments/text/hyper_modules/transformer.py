# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Transformer decoder.
"""
from typing import Callable, Dict, NamedTuple, Optional, Tuple, Union
import warnings

from texar.torch.core import layers
from texar.torch.modules.decoders.decoder_base import _make_output_layer
from texar.torch.modules.decoders.decoder_base import DecoderBase
from texar.torch.modules.decoders.decoder_base import TokenEmbedder
from texar.torch.modules.decoders.decoder_base import TokenPosEmbedder
from texar.torch.modules.decoders.decoder_helpers import EmbeddingHelper
from texar.torch.modules.decoders.decoder_helpers import Helper
from texar.torch.modules.encoders.multihead_attention import Cache
from texar.torch.modules.encoders.multihead_attention import \
    MultiheadAttentionEncoder
from texar.torch.modules.encoders.transformer_encoder import \
    default_transformer_poswise_net_hparams
from texar.torch.modules.networks.networks import FeedForwardNetwork
from texar.torch.utils import transformer_attentions as attn
from texar.torch.utils.beam_search import beam_search
from texar.torch.utils.shapes import mask_sequences
from texar.torch.utils.utils import sequence_mask
import torch
from torch import nn

from src.hyper.layers import get_hyper_layer
from src.hyper.layers import get_hyper_ln_layer

__all__ = [
    'TransformerDecoderOutput',
    'TransformerDecoder',
]

EmbeddingFn = Callable[[torch.LongTensor, torch.LongTensor], torch.Tensor]


class TransformerDecoderOutput(NamedTuple):
    r"""The output of :class:`TransformerDecoder`.
    """
    logits: torch.Tensor
    r"""A :tensor:`Tensor` of shape ``[batch_size, max_time, vocab_size]``
    containing the logits."""
    sample_id: torch.LongTensor
    r"""A :tensor:`LongTensor` of shape ``[batch_size, max_time]``
    (or ``[batch_size, max_time, vocab_size]``) containing the sampled
    token indices. Note that the shape of ``sample_id`` is different for
    different decoding strategy or helper. Please refer to
    :class:`~texar.torch.modules.Helper` for the detailed information."""


class TransformerDecoder(DecoderBase[Cache, TransformerDecoderOutput]):
    # State variables used during `dynamic_decode`. Assigned in `forward`.
    _state_max_decoding_length: int
    _state_context: Optional[torch.LongTensor]
    _state_context_sequence_length: Optional[torch.LongTensor]
    _state_cache: Cache

    def __init__(self,
                 hyper_cfg,
                 token_embedder: Optional[TokenEmbedder] = None,
                 token_pos_embedder: Optional[TokenPosEmbedder] = None,
                 vocab_size: Optional[int] = None,
                 output_layer: Optional[Union[nn.Module, torch.Tensor]] = None,
                 hparams=None):
        super().__init__(
            token_embedder, token_pos_embedder,
            input_time_major=False, output_time_major=False, hparams=hparams)

        if token_pos_embedder is None and token_embedder is not None:
            warnings.warn(
                "Transformer models cannot capture positional information if "
                "no positional embedding is provided.")

        self.hyper_cfg = hyper_cfg
        self._input_size = self._hparams.dim
        self._output_layer, self._vocab_size = _make_output_layer(
            output_layer, vocab_size, self._input_size,
            self._hparams.output_layer_bias)

        self.self_attns = nn.ModuleList()
        self.self_attn_layer_norm = nn.ModuleList()
        self.hyper_self_attn_layer_norm = nn.ModuleList()
        self.enc_dec_attns = nn.ModuleList()
        self.end_dec_attn_layer_norm = nn.ModuleList()
        self.hyper_end_dec_attn_layer_norm = nn.ModuleList()
        self.poswise_networks = nn.ModuleList()
        self.hyper_poswise_networks = nn.ModuleList()
        self.poswise_layer_norm = nn.ModuleList()
        self.hyper_poswise_layer_norm = nn.ModuleList()

        self.initialize_blocks()

        self.final_layer_norm = nn.LayerNorm(self._input_size,
                                             eps=self._hparams.eps)
        self.hyper_final_layer_norm = get_hyper_ln_layer(self._input_size, hyper_cfg)

        self.embed_dropout = nn.Dropout(self._hparams.embedding_dropout)
        self.residual_dropout = nn.Dropout(self._hparams.residual_dropout)

        if self._hparams.initializer:
            # TODO: This might be different to what TensorFlow does
            initialize = layers.get_initializer(self._hparams.initializer)
            assert initialize is not None
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split(".")[-1] == "weight" and "layer_norm" not in name:
                    initialize(param)

    def initialize_blocks(self):
        r"""Helper function which initializes blocks for decoder.

        Should be overridden by any classes where block initialization varies.
        """
        for _ in range(self._hparams.num_blocks):
            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.self_attns.append(attn_module)
            self.self_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))
            self.hyper_self_attn_layer_norm.append(
                get_hyper_ln_layer(self._input_size, self.hyper_cfg))

            attn_module = MultiheadAttentionEncoder(
                self._input_size, self._hparams.multihead_attention)
            if self._hparams.dim != attn_module.output_size:
                raise ValueError("The output dimension of "
                                 "MultiheadEncoder should be equal "
                                 "to the dim of TransformerDecoder")
            self.enc_dec_attns.append(attn_module)
            self.end_dec_attn_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))
            self.hyper_end_dec_attn_layer_norm.append(
                get_hyper_ln_layer(self._input_size, self.hyper_cfg))

            poswise_network = FeedForwardNetwork(
                hparams=self._hparams.poswise_feedforward)
            if (poswise_network.hparams.layers[-1]['kwargs']['out_features']
                    != self._hparams.dim):
                raise ValueError("The output dimension of "
                                 "FeedForwardNetwork should be equal "
                                 "to the dim of TransformerDecoder")
            self.poswise_networks.append(poswise_network)
            self.hyper_poswise_networks.append(get_hyper_layer(256, self.hyper_cfg))
            self.poswise_layer_norm.append(
                nn.LayerNorm(self._input_size, eps=self._hparams.eps))
            self.hyper_poswise_layer_norm.append(
                get_hyper_ln_layer(self._input_size, self.hyper_cfg)
            )

    @staticmethod
    def default_hparams():
        dim = 512
        return {
            'num_blocks': 6,
            'dim': dim,
            'embedding_tie': True,
            'output_layer_bias': False,
            'max_decoding_length': int(1e10),
            'embedding_dropout': 0.1,
            'residual_dropout': 0.1,
            'poswise_feedforward': default_transformer_poswise_net_hparams(dim),
            'multihead_attention': {
                'name': 'multihead_attention',
                'num_units': 512,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'output_dim': 512,
                'use_bias': False,
            },
            'eps': 1e-12,
            'initializer': None,
            'name': "transformer_decoder",
        }

    def _inputs_to_outputs(self, inputs: torch.Tensor,
                           cache: Cache) -> Tuple[torch.Tensor, Cache]:
        outputs = self._self_attention_stack(
            inputs.unsqueeze(1), memory=cache['memory'], cache=cache)
        outputs = self._output_layer(outputs)
        outputs = outputs.squeeze(1)
        return outputs, cache

    def forward(self,  # type: ignore
                inputs: Optional[torch.Tensor] = None,
                sequence_length: Optional[torch.LongTensor] = None,
                memory: Optional[torch.Tensor] = None,
                memory_sequence_length: Optional[torch.LongTensor] = None,
                memory_attention_bias: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                context_sequence_length: Optional[torch.LongTensor] = None,
                helper: Optional[Helper] = None,
                decoding_strategy: str = 'train_greedy',
                max_decoding_length: Optional[int] = None,
                impute_finished: bool = False,
                infer_mode: Optional[bool] = None,
                beam_width: Optional[int] = None,
                length_penalty: float = 0.,
                **kwargs) \
            -> Union[
                TransformerDecoderOutput,
                Tuple[TransformerDecoderOutput, torch.LongTensor],
                Dict[str, torch.Tensor]]:
        if memory is not None:
            if memory_attention_bias is None:
                if memory_sequence_length is None:
                    raise ValueError(
                        "`memory_sequence_length` is required if "
                        "`memory_attention_bias` is not given.")

                enc_padding = 1 - sequence_mask(
                    memory_sequence_length, memory.size(1),
                    dtype=torch.float32)
                memory_attention_bias = attn.attention_bias_ignore_padding(
                    enc_padding)

        # record the context, which will be used in step function
        # for dynamic_decode
        if context is not None:
            if context_sequence_length is None:
                raise ValueError("'context_sequence_length' must not be None"
                                 "when 'context' is specified.")
            self._state_context = context[:, 1:]
            self._state_context_sequence_length = context_sequence_length - 1
        else:
            self._state_context = None
            self._state_context_sequence_length = None

        # Faster code path for teacher-forcing training
        if (helper is None and beam_width is None and
                decoding_strategy == 'train_greedy'):
            if inputs is None:
                raise ValueError("'input' must not be none "
                                 "when using 'train_greedy' decoding strategy.")
            times = torch.arange(
                inputs.size(1), dtype=torch.long, device=inputs.device)
            times = times.unsqueeze(0).expand(inputs.size(0), -1)
            inputs = self.embed_tokens(inputs, times)
            if sequence_length is not None:
                inputs = mask_sequences(inputs, sequence_length)

            decoder_self_attention_bias = (
                attn.attention_bias_lower_triangle(inputs.size(1)))

            decoder_output = self._self_attention_stack(
                inputs, memory, decoder_self_attention_bias,
                memory_attention_bias, cache=None)
            logits = self._output_layer(decoder_output)
            sample_id = torch.argmax(logits, dim=-1)

            return TransformerDecoderOutput(logits, sample_id)

        # Inference code path.
        if max_decoding_length is None:
            max_decoding_length = self._hparams.max_decoding_length

        self._state_max_decoding_length = max_decoding_length

        if beam_width is None or beam_width == 1:  # Inference-like decoding
            # Prepare helper
            if helper is None:
                kwargs.update(decoding_strategy=decoding_strategy)
                if context is not None:
                    kwargs.update(start_tokens=context[:, 0])
                helper = self._create_or_get_helper(infer_mode, **kwargs)
            assert isinstance(helper, EmbeddingHelper)

            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=False, batch_size=helper.batch_size)
            if context is not None:
                assert self._state_context is not None
                pad_length = max_decoding_length - self._state_context.size(1)
                if pad_length > 0:
                    self._state_context = torch.cat((
                        self._state_context,
                        self._state_context.new_zeros(
                            self._state_context.size(0), pad_length)
                    ), dim=1)

            outputs, cache, sequence_lengths = self.dynamic_decode(
                helper, inputs=None, sequence_length=None,
                initial_state=None, max_decoding_length=max_decoding_length,
                impute_finished=impute_finished)
            del cache  # not used

            if context is not None:
                # Here the length of sample_id will be larger than that
                # of logit by 1, because there will be a additional
                # start_token in the returned sample_id.
                # the start_id should be the first token of the
                # given context
                start_tokens = context[:, 0]
                outputs = TransformerDecoderOutput(
                    logits=outputs.logits,
                    sample_id=torch.cat([
                        start_tokens.unsqueeze(1),
                        outputs.sample_id
                    ], dim=1))
                sequence_lengths = sequence_lengths + 1

            return outputs, sequence_lengths

        else:  # Beam-search decoding
            # Ignore `decoding_strategy` and # assume `helper` is not set.
            if helper is not None:
                raise ValueError("Must not set 'beam_width' and 'helper' "
                                 "simultaneously.")
            if context is not None:
                start_tokens = context[:, 0]
            else:
                if 'start_tokens' not in kwargs:
                    raise ValueError(
                        "'start_tokens' must be specified when using"
                        "beam search decoding.")
                start_tokens = kwargs['start_tokens']
            _batch_size = start_tokens.size(0)
            self._state_cache = self._init_cache(
                memory, memory_attention_bias,
                beam_search_decoding=True,
                batch_size=_batch_size)
            end_token: int = kwargs.get('end_token')  # type: ignore

            # The output format is different when running beam search.
            sample_id, log_prob = self.beam_decode(
                start_tokens,
                end_token,
                embedding_fn=self.embed_tokens,
                beam_width=beam_width,
                length_penalty=length_penalty,
                decode_length=max_decoding_length)

            return {
                'sample_id': sample_id,
                'log_prob': log_prob
            }

    def _self_attention_stack(
            self, inputs: torch.Tensor,
            memory: Optional[torch.Tensor],
            decoder_self_attention_bias: Optional[torch.Tensor] = None,
            memory_attention_bias: Optional[torch.Tensor] = None,
            cache: Optional[Cache] = None) -> torch.Tensor:
        r"""Forward through the stacked multi-head attentions.
        """
        inputs = self.embed_dropout(inputs)
        if cache is not None:
            if memory is not None:
                memory_attention_bias = cache['memory_attention_bias']
        else:
            assert decoder_self_attention_bias is not None

        x = inputs
        for i in range(self._hparams.num_blocks):
            layer_cache = cache['layers'][i] if cache is not None else None

            # TODO(JB): Changed LN here.
            # selfatt_output = self.self_attns[i](
            #     queries=self.self_attn_layer_norm[i](x),
            #     memory=None,
            #     memory_attention_bias=decoder_self_attention_bias,
            #     cache=layer_cache)
            selfatt_output = self.self_attns[i](
                queries=self.hyper_self_attn_layer_norm[i](x),
                memory=None,
                memory_attention_bias=decoder_self_attention_bias,
                cache=layer_cache)
            x = x + self.residual_dropout(selfatt_output)

            if memory is not None:
                # TODO(JB): Changed LN here.
                # encdec_output = self.enc_dec_attns[i](
                #     queries=self.end_dec_attn_layer_norm[i](x),
                #     memory=memory,
                #     memory_attention_bias=memory_attention_bias)
                encdec_output = self.enc_dec_attns[i](
                    queries=self.hyper_end_dec_attn_layer_norm[i](x),
                    memory=memory,
                    memory_attention_bias=memory_attention_bias)
                x = x + self.residual_dropout(encdec_output)

            # TODO(JB): Changed MLP here.
            # sub_output = self.poswise_networks[i](self.poswise_layer_norm[i](x))
            sub_output = self.poswise_networks[i](self.hyper_poswise_layer_norm[i](x))
            sub_output = self.hyper_poswise_networks[i](sub_output)
            x = x + self.residual_dropout(sub_output)

        # TODO(JB): Changed LN here.
        # return self.final_layer_norm(x)
        return self.hyper_final_layer_norm(x)

    def _init_cache(self, memory: Optional[torch.Tensor],
                    memory_attention_bias: Optional[torch.Tensor],
                    beam_search_decoding: bool,
                    batch_size: int) -> Cache:
        r"""Returns an initialized cache.

        In order to support both inference-like decoding and beam-search
        decoding, the elements of each layer must be initialized and extended
        as different structure respectively. Specifically, for inference-like
        decoding, a simple list is used; for beam-search decoding, a
        :tensor:`Tensor` of shape ``[batch_size, current_steps, num_units]``
        is maintained, where ``current_steps`` is the number of steps currently
        decoded.
        """

        device = next(self.parameters()).device

        def _create_ta():
            return []

        def _create_empty_tensor():
            ret = torch.zeros(
                batch_size, 0, self._hparams.multihead_attention.num_units,
                dtype=torch.float, device=device)
            return ret

        _create_fn = (_create_empty_tensor if beam_search_decoding
                      else _create_ta)

        cache: Cache = {
            'memory': memory,
            'memory_attention_bias': memory_attention_bias,
            'layers': [{
                'keys': _create_fn(),
                'values': _create_fn(),
            } for _ in range(self._hparams.num_blocks)],
        }

        return cache

    def beam_decode(self, start_tokens: torch.LongTensor, end_token: int,
                    embedding_fn: Callable[
                        [torch.LongTensor, torch.LongTensor], torch.Tensor],
                    decode_length: int = 256, beam_width: int = 5,
                    length_penalty: float = 0.6) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        def _symbols_to_logits_fn(ids, cache):
            batch_size = ids.size(0)
            step = ids.size(-1) - 1
            times = ids.new_full((batch_size,), step)
            inputs = embedding_fn(ids[:, -1], times)
            return self._inputs_to_outputs(inputs, cache)

        assert self._vocab_size is not None

        outputs, log_prob = beam_search(
            _symbols_to_logits_fn,
            start_tokens,
            beam_width,
            decode_length,
            self._vocab_size,
            length_penalty,
            states=self._state_cache,
            eos_id=end_token)

        # Ignores <BOS>
        outputs = outputs[:, :, 1:]
        # shape = [batch_size, seq_length, beam_width]
        outputs = outputs.permute(0, 2, 1)
        return outputs, log_prob

    @property
    def output_size(self) -> int:
        r"""Output size of one step.
        """
        return self._input_size

    def initialize(self, helper: Helper, inputs: Optional[torch.Tensor],
                   sequence_length: Optional[torch.LongTensor],
                   initial_state: Optional[Cache]) \
            -> Tuple[torch.ByteTensor, torch.Tensor, Cache]:
        initial_finished, initial_inputs = helper.initialize(
            self.embed_tokens, inputs, sequence_length)
        state = initial_state or self._state_cache
        return initial_finished, initial_inputs, state

    def step(self, helper: Helper, time: int, inputs: torch.Tensor,
             state: Optional[Cache]) -> \
            Tuple[TransformerDecoderOutput, Cache]:
        assert state is not None
        outputs, state = self._inputs_to_outputs(inputs, state)
        sample_ids = helper.sample(time=time, outputs=outputs)
        if self._state_context is not None:
            assert self._state_context_sequence_length is not None
            sample_ids = torch.where(
                self._state_context_sequence_length > time,
                self._state_context[:, time],
                sample_ids)

        next_state = state
        outputs = TransformerDecoderOutput(
            logits=outputs,
            sample_id=sample_ids)
        return outputs, next_state

    def next_inputs(self, helper: Helper, time: int,
                    outputs: TransformerDecoderOutput) -> \
            Tuple[torch.Tensor, torch.ByteTensor]:
        finished, next_inputs = helper.next_inputs(
            self.embed_tokens, time, outputs.logits, outputs.sample_id)
        return next_inputs, finished

    def finalize(self,  # type: ignore
                 outputs: TransformerDecoderOutput,
                 final_state: Optional[Cache],
                 sequence_lengths: torch.LongTensor) \
            -> Tuple[TransformerDecoderOutput, Optional[Cache]]:
        # Clear state variables at end of decoding.
        del self._state_max_decoding_length
        del self._state_context
        del self._state_context_sequence_length
        del self._state_cache

        return super().finalize(outputs, final_state, sequence_lengths)
