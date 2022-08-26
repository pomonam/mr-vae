import torch.optim

from workloads.spec import Workload
from workloads.binary_image.input_pipeline import load_data

from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST as Encoder_VAE
from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_AE_MNIST as Decoder_AE
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers.training_callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)


class BinaryImageWorkload(Workload):
  _AVAIL_DATA = ["omniglot", "mnist"]
  _AVAIL_ARCH = ["vae", "vq_vae"]

  def __init__(self, train_cfg, data_name, architecture):
    self.train_cfg = train_cfg
    self.data_name = data_name
    self.architecture = architecture

    self.model = self.init_model()
    self.optimizer = self.init_optimizer()
    self.scheduler = self.init_scheduler()
    self.train_loader, self.eval_loader = self.init_loaders()

    callbacks = [TrainingCallback()]
    self.callback_handler = CallbackHandler(
      callbacks=callbacks, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler
    )

    self.callback_handler.add_callback(ProgressBarCallback())
    self.callback_handler.add_callback(MetricConsolePrinterCallback())

  # def _set_inputs_to_device(self, inputs):
  #
  #   inputs_on_device = inputs
  #
  #   if self.device == "cuda":
  #     cuda_inputs = dict.fromkeys(inputs)
  #
  #     for key in inputs.keys():
  #       if torch.is_tensor(inputs[key]):
  #         cuda_inputs[key] = inputs[key].cuda()
  #
  #       else:
  #         cuda_inputs = inputs[key]
  #     inputs_on_device = cuda_inputs
  #
  #   return inputs_on_device

  def init_model(self):
    model_config = BetaVAEConfig()
    model_config.latent_dim = self.latent_dim

    model = BetaVAE(
      model_config=model_config,
      encoder=Encoder_VAE(model_config),
      decoder=Decoder_AE(model_config),
    )
    return model

  def init_optimizer(self):
    optimizer = torch.optim.Adam(self.model.parameters(), self.train_cfg.lr)
    self.optimizer = optimizer
    return optimizer

  def init_scheduler(self):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
      self.optimizer, milestones=[200, 350, 500, 750, 1000], gamma=10 ** (-1 / 5), verbose=True
    )
    self.scheduler = scheduler
    return scheduler

  def init_loaders(self):
    if self.data_name == "mnist":
      train_loader = load_data("train", self.batch_size)
      eval_loader = load_data("test", self.batch_size)
    else:
      pass
    return train_loader, eval_loader

  def init_data(self):
    pass

  def train_step(self, epoch):
    self.callback_handler.on_train_step_begin(
      training_config=self.train_cfg,
      train_loader=self.train_loader,
      epoch=epoch,
    )
    self.model.train()

    epoch_loss = 0

    for inputs in self.train_loader:
      model_output = self.model(
        inputs, epoch=epoch, dataset_size=len(self.train_loader.dataset)
      )
      loss = model_output.loss
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      epoch_loss = epoch_loss + loss.item()

      if epoch_loss != epoch_loss:
        raise ArithmeticError("NaN detected in train loss")

      self.callback_handler.on_train_step_end(
        training_config=self.training_config
      )

    self.model.update()
    epoch_loss = epoch_loss / len(self.train_loader)
    return epoch_loss

  def eval_step(self, epoch):
    self.callback_handler.on_eval_step_begin(
      training_config=self.training_config,
      eval_loader=self.eval_loader,
      epoch=epoch,
    )

    self.model.eval()

    epoch_loss = 0

    for inputs in self.eval_loader:

      with torch.no_grad():
        model_output = self.model(
          inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
        )
        loss = model_output.loss

        epoch_loss += loss.item()

        if epoch_loss != epoch_loss:
          raise ArithmeticError("NaN detected in eval loss")

        self.callback_handler.on_eval_step_end(training_config=self.training_config)

    epoch_loss /= len(self.eval_loader)

    return epoch_loss

  def scheduler_step(self, metrics=None):
    if metrics is not None:
      self.scheduler.step(metrics)
    else:
      self.scheduler.step()

  @property
  def batch_size(self):
    return 128

  @property
  def num_epochs(self):
    return 1000

  @property
  def latent_dim(self):
    return 32

  @property
  def save_interval(self):
    return 500

  @property
  def predict_interval(self):
    return 500

  @property
  def reconsturction_loss(self):
    return "mse"

  @property
  def predict_inteval(self):
    return 500

  def predict(self):

    self.model.eval()

    # with torch.no_grad():

    inputs = self.eval_loader.dataset[
             : min(self.eval_loader.dataset.data.shape[0], 10)
             ]
    # inputs = self._set_inputs_to_device(inputs)

    model_out = self.model(inputs)
    reconstructions = model_out.recon_x.cpu().detach()
    z_enc = model_out.z
    z = torch.randn_like(z_enc)
    normal_generation = self.model.decoder(z).reconstruction.detach().cpu()

    return inputs["data"], reconstructions, normal_generation


  def save_checkpoint(self, dir_path, epoch: int):
    """Saves a checkpoint alowing to restart training from here
    Args:
        dir_path (str): The folder where the checkpoint should be saved
        epochs_signature (int): The epoch number"""

    checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    # save optimizer
    torch.save(
      deepcopy(self.optimizer.state_dict()),
      os.path.join(checkpoint_dir, "optimizer.pt"),
    )

    # save scheduler
    torch.save(
      deepcopy(self.scheduler.state_dict()),
      os.path.join(checkpoint_dir, "scheduler.pt"),
    )

    # save model
    self.model.save(checkpoint_dir)

    # save training config
    self.train_cfg.save_json(checkpoint_dir, "training_config")


  def save_model(self, dir_path: str):
      """This method saves the final model along with the config files
      Args:
          model (BaseAE): The model to be saved
          dir_path (str): The folder where the model and config files should be saved
      """

      if not os.path.exists(dir_path):
          os.makedirs(dir_path)

      # save model
      self.model.save(dir_path)

      # save training config
      self.train_cfg.save_json(dir_path, "training_config")
