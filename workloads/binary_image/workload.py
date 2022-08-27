import torch.optim

from workloads.spec import Workload
from workloads.binary_image.input_pipeline import load_data, build_input_queue
from pythae.trainers.training_callbacks import WandbCallback
from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST as Encoder_VAE
from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_AE_MNIST as Decoder_AE
from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers.training_callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
import os
from copy import deepcopy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BinaryImageWorkload(Workload):
  _AVAIL_DATA = ["omniglot", "mnist"]
  _AVAIL_ARCH = ["vae", "vq_vae"]

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
  def reconsturction_loss(self):
    return "mse"

  @property
  def predict_interval(self):
    return 3

  @property
  def save_interval(self):
    return 500

  def __init__(self, train_cfg, data_name, architecture):
    self.train_cfg = train_cfg
    self.data_name = data_name
    self.architecture = architecture

    self.model = self.init_model()
    self.optimizer = self.init_optimizer()
    self.scheduler = self.init_scheduler()
    self.train_loader, self.eval_loader = self.init_loaders()

    callbacks = [TrainingCallback()]
    wandb_cb = WandbCallback()
    project_name = "binary_image" + "_" + self.data_name + "_" + self.architecture
    wandb_cb.setup(
      training_config=self.train_cfg,
      project_name=project_name,
      entity_name="bae-group"
    )
    callbacks.append(wandb_cb)
    self.callback_handler = CallbackHandler(
      callbacks=callbacks, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler
    )

    self.callback_handler.add_callback(ProgressBarCallback())
    # self.callback_handler.add_callback(MetricConsolePrinterCallback())

  def init_model(self):
    model_config = BetaVAEConfig()
    model_config.latent_dim = self.latent_dim
    model_config.reconstruction_loss = self.reconsturction_loss
    model = BetaVAE(
      model_config=model_config,
      encoder=Encoder_VAE(model_config),
      decoder=Decoder_AE(model_config),
    )
    return model.to(DEVICE)

  def init_optimizer(self):
    optimizer = torch.optim.Adam(self.model.parameters(), self.train_cfg.lr)
    self.optimizer = optimizer
    return optimizer

  def init_scheduler(self):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
      self.optimizer, milestones=[200, 350, 500, 750, 1000], gamma=10 ** (-1 / 5)
    )
    self.scheduler = scheduler
    return scheduler

  def init_loaders(self):
    if self.data_name == "mnist":
      train_loader = load_data("train", self.batch_size)
      eval_loader = load_data("test", self.batch_size)
    elif self.data_name == "omniglot":
      pass
    else:
      raise NotImplementedError
    return train_loader, eval_loader

  def train_step(self, epoch):
    self.callback_handler.on_train_step_begin(
      training_config=self.train_cfg,
      train_loader=self.train_loader,
      epoch=epoch,
    )
    self.model.train()

    epoch_loss = 0

    queue = build_input_queue(self.train_loader, DEVICE)
    for inputs in queue:
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
        training_config=self.train_cfg
      )

    self.model.update()
    epoch_loss = epoch_loss / len(self.train_loader)
    return epoch_loss

  def eval_step(self, epoch):
    self.callback_handler.on_eval_step_begin(
      training_config=self.train_cfg,
      eval_loader=self.eval_loader,
      epoch=epoch,
    )

    self.model.eval()

    epoch_loss = 0

    queue = build_input_queue(self.eval_loader, DEVICE)
    for inputs in queue:

      with torch.no_grad():
        model_output = self.model(
          inputs, epoch=epoch, dataset_size=len(self.eval_loader.dataset)
        )
        loss = model_output.loss

        epoch_loss += loss.item()

        if epoch_loss != epoch_loss:
          raise ArithmeticError("NaN detected in eval loss")

        self.callback_handler.on_eval_step_end(training_config=self.train_cfg)

    epoch_loss /= len(self.eval_loader)

    return epoch_loss

  def scheduler_step(self, metrics=None):
    # if metrics is not None:
    #   self.scheduler.step(metrics)
    # else:
    #   self.scheduler.step()
    self.scheduler.step()

  def predict(self):
    self.model.eval()
    inputs = {"data": iter(self.train_loader).next()[:10].reshape(-1, 1, 28, 28).to(DEVICE)}
    model_out = self.model(inputs)
    reconstructions = model_out.recon_x.cpu().detach()
    z_enc = model_out.z
    z = torch.randn_like(z_enc)
    normal_generation = self.model.decoder(z).reconstruction.detach().cpu()
    return inputs["data"], reconstructions, normal_generation

