from workloads.spec import Workload

from pythae.models.nn.benchmarks.cifar import Encoder_ResNet_AE_CIFAR as Encoder_AE
from pythae.models.nn.benchmarks.cifar import Encoder_ResNet_VAE_CIFAR as Encoder_VAE
from pythae.models.nn.benchmarks.cifar import Encoder_ResNet_SVAE_CIFAR as Encoder_SVAE
from pythae.models.nn.benchmarks.cifar import Encoder_ResNet_VQVAE_CIFAR as Encoder_VQVAE
from pythae.models.nn.benchmarks.cifar import Decoder_ResNet_AE_CIFAR as Decoder_AE
from pythae.models.nn.benchmarks.cifar import Decoder_ResNet_VQVAE_CIFAR as Decoder_VQVAE
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.models import VQVAE, VQVAEConfig


class ImageWorkload(Workload):

  def __init__(self, data_name, architecture):
    self.data_name = data_name

  def init_model_fn(self):
    model_config = BetaVAEConfig()

    model = BetaVAE(
      model_config=model_config,
      encoder=Encoder_VAE(model_config),
      decoder=Decoder_AE(model_config),
    )
    return model



  @property
  def batch_size(self):
    return 128

  @property
  def num_epochs(self):
    return 500

  @property
  def latent_dim(self):
    return 32

  @property
  def reconsturction_loss(self):
    return "mse"
