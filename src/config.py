from typing import Optional

from src.schedules import build_betas_schedule


def get_ns(args: dict, name: str) -> Optional:
  if name in args:
    return getattr(args, name)
  return None


class TrainConfig:

  def __init__(self, args: dict) -> None:
    self.total_epochs = get_ns(args, "total_epochs")
    self.warmup_epochs = get_ns(args, "warmup_epochs")

    self.lr = get_ns(args, "lr")
    self.batch_size = get_ns(args, "batch_size")
    self.beta = get_ns(args, "beta")
    self.schedule = get_ns(args, "schedule")

    self.seed = get_ns(args, "seed")
    self.checkpoint_dir = get_ns(args, "checkpoint_dir")
    self.save_freq = get_ns(args, "save_freq")
    self.eval_freq = get_ns(args, "eval_freq")

    if self.schedule is not None and self.beta is not None:
      self.beta_schedule = build_betas_schedule(self.schedule,
                                                self.beta,
                                                self.total_epochs)
    else:
      self.beta_schedule = None

  def get_beta(self, epoch):
    if self.beta_schedule is not None:
      return self.beta_schedule[epoch]


class HyperConfig:
  non_shared_emd_dim = 16 * 4
  shared_preprocess_dim = 64

  def __init__(self, args: dict) -> None:
    self.shared_preprocess = get_ns(args, "shared_preprocess")

    self.apply_zero_init = get_ns(args, "apply_zero_init")

    self.param_type = get_ns(args, "param_type")
    self.layer_type = get_ns(args, "layer_type")
    self.block_type = get_ns(args, "block_type")
    self.preprocess_beta = get_ns(args, "preprocess_beta")
    self.include_hyper_bn = get_ns(args, "include_hyper_bn")
    self.include_output_stem = get_ns(args, "include_output_stem")
    self.include_latent_stem = get_ns(args, "include_latent_stem")
