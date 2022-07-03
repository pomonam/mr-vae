
"""Variational Autoencoder example on binarized MNIST dataset."""

from functools import partial
import os
from typing import (Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple,
                    Union)

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.10'
jax.config.update('jax_platform_name', 'cpu')

PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)


def load_dataset(split: str, batch_size: int) -> Iterator[Batch]:
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True,
                 read_config=tfds.ReadConfig(shuffle_seed=FLAGS.random_seed))
  ds = ds.shuffle(buffer_size=10 * batch_size, seed=FLAGS.random_seed)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return iter(tfds.as_numpy(ds))


class Emb_Like(hk.Module):
  def __init__(
          self,
          vocab_size: Optional[int] = None,
          embed_dim: Optional[int] = None,
          embedding_matrix: Optional[jnp.ndarray] = None,
          w_init: Optional[hk.initializers.Initializer] = None,
          lookup_style: Union[str, hk.EmbedLookupStyle] = "ARRAY_INDEX",
          name: Optional[str] = None,
          precision: jax.lax.Precision = jax.lax.Precision.HIGHEST,
  ):
    super().__init__(name=name)
    if embedding_matrix is not None:
      embedding_matrix = jnp.asarray(embedding_matrix)
      w_init = lambda _, __: embedding_matrix
      vocab_size = embedding_matrix.shape[0]
      embed_dim = embedding_matrix.shape[1]

    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.lookup_style = lookup_style
    self.precision = precision
    # self.w_init = w_init or hk.initializers.TruncatedNormal()
    self.w_init = w_init or jnp.ones

  @property
  def embeddings(self):
    if not self.embed_dim:
      raise ValueError(
        "Emb_Like could not initial `embeddings` "
        "without `embed_dim`.")
    return hk.get_parameter("embeddings", [self.vocab_size, self.embed_dim],
                            init=self.w_init)

  def __call__(
          self,
          inputs: jnp.ndarray,
          ids: jnp.ndarray,
          lookup_style: Optional[Union[str, hk.EmbedLookupStyle]] = None,
          precision: Optional[jax.lax.Precision] = None,
  ) -> jnp.ndarray:

    ## always assume channel last
    embed_dim = self.embed_dim = inputs.shape[-1]
    ## tile according to the batch_size
    batch_size = inputs.shape[0]
    num_models = ids.shape[0]

    if num_models > batch_size:
      ids = ids[:batch_size]

    if batch_size > num_models and (batch_size % num_models) != 0:
      raise ValueError(
        f"batch_size {batch_size} and "
        f"number of models {num_models} do not divide.")
    if batch_size > num_models:
      ids = jnp.tile(ids, batch_size // num_models)

    # TODO(tomhennigan) Consider removing asarray here.
    ids = jnp.asarray(ids)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
      raise ValueError("hk.Embed's __call__ method must take an array of "
                       "integer dtype but was called with an array of "
                       f"{ids.dtype}")

    lookup_style = lookup_style or self.lookup_style
    if isinstance(lookup_style, str):
      lookup_style = getattr(hk.EmbedLookupStyle, lookup_style.upper())

    if lookup_style == hk.EmbedLookupStyle.ARRAY_INDEX:
      # If you don't wrap ids in a singleton tuple then JAX will try to unpack
      # it along the row dimension and treat each row as a separate index into
      # one of the dimensions of the array. The error only surfaces when
      # indexing with DeviceArray, while indexing with numpy.ndarray works fine.
      # See https://github.com/google/jax/issues/620 for more details.
      # Cast to a jnp array in case `ids` is a tracer (eg un a dynamic_unroll).
      return jnp.asarray(self.embeddings)[(ids,)]

    elif lookup_style == hk.EmbedLookupStyle.ONE_HOT:
      one_hot_ids = jax.nn.one_hot(ids, self.vocab_size)
      precision = self.precision if precision is None else precision
      return jnp.dot(one_hot_ids, self.embeddings, precision=precision)

    else:
      raise NotImplementedError(f"{lookup_style} is not supported by hk.Embed.")


class CausalSelfAttention(hk.MultiHeadAttention):
  """Self attention with a causal mask applied."""

  def __call__(
      self,
      query: jnp.ndarray,
      key: Optional[jnp.ndarray] = None,
      value: Optional[jnp.ndarray] = None,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    key = key if key is not None else query
    value = value if value is not None else query

    if query.ndim != 3:
      raise ValueError('Expect queries of shape [B, T, D].')

    seq_len = query.shape[1]
    causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
    mask = mask * causal_mask if mask is not None else causal_mask

    return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               init_scale: float,
               widening_factor: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=initializer)(x)


class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               dropout_rate: float,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate

  def __call__(self,
               h: jnp.ndarray,
               mask: Optional[jnp.ndarray],
               ids,
               is_training: bool) -> jnp.ndarray:
    """Connects the transformer.
    Args:
      h: Inputs, [B, T, D].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.
    Returns:
      Array of shape [B, T, D].
    """

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    if mask is not None:
      mask = mask[:, None, None, :]

    # Note: names chosen to approximately match those used in the GPT-2 code;
    # see https://github.com/openai/gpt-2/blob/master/src/model.py.
    for i in range(self._num_layers):
      h_norm = layer_norm(h, name=f'h{i}_ln_1')
      #
      # print(h_norm.shape)
      h_norm = Emb_Like(FLAGS.batch_size, )(h_norm, ids)[:,jnp.newaxis,:] * h_norm
      #
      h_attn = CausalSelfAttention(
          num_heads=self._num_heads,
          key_size=32,
          model_size=h.shape[-1],
          w_init_scale=init_scale,
          name=f'h{i}_attn')(h_norm, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_2')
      #
      # print(h_norm.shape)
      h_norm = Emb_Like(FLAGS.batch_size, )(h_norm, ids)[:,jnp.newaxis,:] * h_norm
      #
      h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = layer_norm(h, name='ln_f')
    #
    # print(h.shape)
    h = Emb_Like(FLAGS.batch_size, )(h, ids)[:,jnp.newaxis,:] * h
    #
    return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a unique LayerNorm to x with default settings."""
  return hk.LayerNorm(axis=-1,
                      create_scale=True,
                      create_offset=True,
                      name=name)(x)

class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size: int = 512, latent_size: int = 10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x: jnp.ndarray, ids) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = hk.Flatten()(x)
    x = Emb_Like(FLAGS.batch_size, )(x, ids) * x
    x = hk.Linear(self._hidden_size)(x)
    x = Emb_Like(FLAGS.batch_size, )(x, ids) * x
    x = jax.nn.relu(x)

    ##
    x = Emb_Like(FLAGS.batch_size, )(x, ids) * x
    #
    mean = hk.Linear(self._latent_size)(x)
    ##
    mean = Emb_Like(FLAGS.batch_size, )(mean, ids) * mean
    #
    log_stddev = hk.Linear(self._latent_size)(x)
    ##
    log_stddev = Emb_Like(FLAGS.batch_size, )(log_stddev, ids) * log_stddev
    #
    stddev = jnp.exp(log_stddev)

    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(
      self,
      hidden_size: int = 512,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(np.prod(self._output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class TransformerDecoder(hk.Module):
  """Decoder model."""

  def __init__(
          self,
          hidden_size: int = 128,
          num_heads: int = 3,
          num_layers: int = 4,
          output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._num_heads = num_heads
    self._num_layers = num_layers
    self._output_shape = output_shape

  def __call__(self, z: jnp.ndarray, x: jnp.ndarray, ids, sampling=False) -> jnp.ndarray:
    x_shape = x.shape
    seq_length = 28 * 28
    vocab_size = 2
    x_flatten = x.reshape((x_shape[0], seq_length))
    x_flatten = jnp.concatenate((jnp.zeros((x_shape[0], 1)), x_flatten[:, :-1]), 1)
    input_mask = jnp.greater(x_flatten, -1)

    # z = jnp.zeros((x_shape[0], 128))

    # Embed the input tokens and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, self._hidden_size, w_init=embed_init)
    transformer = Transformer(
      num_heads=self._num_heads, num_layers=self._num_layers, dropout_rate=0)
    output_layer = hk.Linear(1)
    positional_embeddings = hk.get_parameter(
      'pos_embs', [seq_length, self._hidden_size], init=embed_init)
    z_layer = hk.Linear(self._hidden_size)

    if sampling == False:
      token_embs = token_embedding_map(jnp.int32(x_flatten))
      # transformed_z = z_layer(z).reshape(x_shape[0], 1, 128)
      transformed_z = z.reshape(x_shape[0], 1, 128)
      token_embs = jax.lax.dynamic_update_slice(token_embs, transformed_z, (0, 0, 0))
      input_embeddings = token_embs + positional_embeddings
      output_embeddings = transformer(input_embeddings, input_mask,
                                      ids,
                                      is_training=False)
      logits = output_layer(output_embeddings)
    else:
      def scan_f(prev_state, inputs):
        input_tokens, cur_len = prev_state
        # print(input_tokens.shape)
        ### conditional
        # input_tokens = x_flatten
        input_tokens_cond = input_tokens
        # input_tokens = x_flatten
        # x_flatten_sliced = jax.lax.dynamic_slice(x_flatten, (0, 0), (x_shape[0], 300))
        # input_tokens_cond = jax.lax.dynamic_update_slice(input_tokens, x_flatten_sliced, (0, 1))
        ###
        token_embs = token_embedding_map(jnp.int32(input_tokens_cond))
        # transformed_z = z_layer(z).reshape(x_shape[0], 1, 128)
        transformed_z = z.reshape(x_shape[0], 1, 128)
        token_embs = jax.lax.dynamic_update_slice(token_embs, transformed_z, (0, 0, 0))
        # print(token_embs.shape)
        input_embeddings = token_embs + positional_embeddings
        output_embeddings = transformer(input_embeddings, input_mask,
                                        is_training=False)
        logits = output_layer(output_embeddings)
        # print(logits.shape)
        p = jax.nn.sigmoid(logits)
        outputs = jax.random.bernoulli(hk.next_rng_key(), p)
        outputs = jnp.float32(outputs.reshape(x_flatten.shape))

        # outputs = outputs[:, cur_len:cur_len+1]
        outputs = jax.lax.dynamic_slice(outputs, (0, cur_len), (x_shape[0], 1))
        new_input_tokens = jax.lax.dynamic_update_slice(input_tokens, outputs, (0, cur_len + 1))

        next_state = (new_input_tokens, cur_len + 1)
        return next_state, outputs

      initial_state = (jnp.zeros(x_flatten.shape), 0)
      carry, outputs = hk.scan(scan_f, initial_state, xs=None, length=seq_length)
      # logits = carry[0]
      logits = outputs.transpose((1, 0, 2))
      # logits = x_flatten

    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits

class VAEOutput(NamedTuple):
  image: jnp.ndarray
  mean: jnp.ndarray
  stddev: jnp.ndarray
  logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(
      self,
      hidden_size: int = 1024,
      latent_size: int = 128,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape

  def __call__(self, x: jnp.ndarray, ids, sampling=False) -> VAEOutput:
    x = x.astype(jnp.float32)
    mean, stddev = Encoder(self._hidden_size, self._latent_size)(x, ids)
    if sampling==False:
      z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    else:
      z = jax.random.normal(hk.next_rng_key(), mean.shape)
    # logits = Decoder(self._hidden_size, self._output_shape)(z)
    decoder = TransformerDecoder(output_shape=self._output_shape)
    logits = decoder(z, x, ids, sampling=sampling)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, mean, stddev, logits)


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  """Calculate binary (logistic) cross-entropy from distribution logits.
  Args:
    x: input variable tensor, must be of same shape as logits
    logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))
  Returns:
    A scalar representing binary CE for the given Bernoulli distribution.
  """
  if x.shape != logits.shape:
    raise ValueError("inputs x and logits must be of the same shape")

  x = jnp.reshape(x, (x.shape[0], -1))
  logits = jnp.reshape(logits, (logits.shape[0], -1))

  return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.
  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution
  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


# @partial(jax.vmap, in_axes=(0, None))
def get_beta(ids, beta_list):
  return beta_list[ids%len(beta_list)]


def loss_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch, ids, iteration) -> jnp.ndarray:
  """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs: VAEOutput = model.apply(params, rng_key, batch["image"], ids)

  log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  # beta = beta_schedule_fn(iteration)
  beta = get_beta(ids, beta_list)[:len(log_likelihood)]
  elbo = log_likelihood - beta*kl

  return -jnp.mean(elbo), (-log_likelihood.mean(), kl.mean())

# @jax.jit
def update(
    params: hk.Params,
    rng_key: PRNGKey,
    opt_state: optax.OptState,
    batch: Batch,
    ids,
    iteration,
) -> Tuple[hk.Params, optax.OptState]:
  """Single SGD update step."""
  (val, (recon, kl)), grads  = jax.value_and_grad(loss_fn, has_aux=True)(params, rng_key, batch, ids, iteration)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, new_opt_state, (val, recon, kl)

@jax.jit
def sampling(
    params: hk.Params,
    rng_key: PRNGKey,
    batch: Batch,
):
  outputs = model.apply(params, rng_key, x=batch['image'], sampling=True)
  return outputs[3]


config = {
    "batch_size": 16,
    "learning_rate": 0.001,
    "training_steps": 5000,
    "eval_frequency": 100,
    "random_seed": 42,
    # "beta": 0.5,
    "beta": 1.,
}

print(config)
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
FLAGS = dotdict(config)

beta_schedule_fn = optax.linear_schedule(
        init_value=0.001,
        # init_value=FLAGS.beta,
        end_value=FLAGS.beta,
        transition_steps=1500,
        )

# beta_list = jnp.array([1e-4, 1e-3, 1e-2, 1e-1, 1., 2., 4., 8.])
# beta_list = jnp.array([5e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.,])

beta_list = jnp.array([1e-2, 2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1., 2.5,])
# beta_list = jnp.array([2.5e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1., 2.5, 5.])


def forward(x, ids, sampling=False):
  return VariationalAutoEncoder()(x, ids, sampling=sampling)


model = hk.transform(forward)  # pylint: disable=unnecessary-lambda
optimizer = optax.adam(FLAGS.learning_rate)

rng_seq = hk.PRNGSequence(FLAGS.random_seed)
params = model.init(next(rng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)), jnp.arange(1))
opt_state = optimizer.init(params)

train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

ids = jnp.arange(FLAGS.batch_size)
# ids_test = jnp.ones(FLAGS.batch_size, dtype=jnp.int32)*4

for step in range(FLAGS.training_steps):
  params, opt_state, losses = update(params, next(rng_seq), opt_state, next(train_ds), ids, opt_state[-2].count.item())

  if step % FLAGS.eval_frequency == 0:
    eval_train_rng_seq, eval_train_ds = next(rng_seq), next(train_ds)
    eval_rng_seq, eval_valid_ds = next(rng_seq), next(valid_ds)
    for i in range(len(beta_list)):
      ids_test = jnp.ones(FLAGS.batch_size, dtype=jnp.int32) * i
      train_loss, (train_recon, train_kl) = loss_fn(params, eval_train_rng_seq, eval_train_ds, ids_test,
                                                    opt_state[-2].count.item())
      print("STEP: {:}; Train ELBO: {:.3f}; Train Recon: {:.1f}; Train KL: {:.1f}; beta: {:.2f}".format(step,
                                                                                                        -train_loss,
                                                                                                        train_recon,
                                                                                                        train_kl,
                                                                                                        beta_list[i]))
      val_loss, (val_recon, val_kl) = loss_fn(params, eval_rng_seq, eval_valid_ds, ids_test, opt_state[-2].count.item())
      print("STEP: {:}; Val ELBO: {:.3f}; Val Recon: {:.1f}; Val KL: {:.1f}; beta: {:.2f}".format(step,
                                                                                                  -val_loss, val_recon,
                                                                                                  val_kl, beta_list[i]))