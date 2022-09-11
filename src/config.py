from argparse import Namespace
from typing import Optional

from src.schedules import build_betas_schedule


def get_ns(args: Namespace, name: str) -> Optional:
  if name in args:
    return getattr(args, name)
  return None


class TrainConfig:

  def __init__(self, args: Namespace) -> None:
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

  def get_beta(self, epoch: int) -> float:
    if self.beta_schedule is not None:
      return self.beta_schedule[epoch]


class HyperConfig:
  non_shared_emd_dim = 16 * 4
  shared_preprocess_dim = 64

  def __init__(self, args: Optional[Namespace]) -> None:
    if args is not None:
      self.hyper_config_summary = get_ns(args, "hyper_config_summary")
      if self.hyper_config_summary is not None:
        if "lin" in self.hyper_config_summary:
          self.shared_preprocess = 0
          self.apply_zero_init = 0
          self.reduce_range = 1
          self.reduce_range = 1

          self.param_type = "post_act"
          self.layer_type = "sig_gate"
          self.block_type = "linear"
          self.norm_type = "scale_shift"

        elif "smlp" in self.hyper_config_summary:
          self.shared_preprocess = 1
          self.apply_zero_init = 0
          self.reduce_range = 1
          self.reduce_range = 1

          self.param_type = "post_act"
          self.layer_type = "sig_gate"
          self.block_type = "mlp"
          self.norm_type = "scale_shift"

        elif "aff" in self.hyper_config_summary:
          self.shared_preprocess = 1
          self.apply_zero_init = 0
          self.reduce_range = 1
          self.reduce_range = 1

          self.param_type = "post_act"
          self.layer_type = "affine"
          self.block_type = "mlp"
          self.norm_type = "scale_shift"

        else:
          raise NotImplementedError()

        if "_bn" in self.hyper_config_summary:
          self.apply_bn_tracking = 1
          self.apply_bn_calibrate = 0
          self.apply_bn_replace = 0

        elif "in" in self.hyper_config_summary:
          self.apply_bn_tracking = 1
          self.apply_bn_calibrate = 0
          self.apply_bn_replace = 1

        else:
          raise NotImplementedError()

        if "_stem" in self.hyper_config_summary:
          self.include_encoder_stem = 1
          self.include_decoder_stem = 1

        else:
          self.include_encoder_stem = 0
          self.include_decoder_stem = 0

      else:
        self.shared_preprocess = get_ns(args, "shared_preprocess")
        self.apply_zero_init = get_ns(args, "apply_zero_init")
        self.apply_bn_tracking = get_ns(args, "apply_bn_tracking")
        self.apply_bn_calibrate = get_ns(args, "apply_bn_calibrate")
        self.apply_bn_replace = get_ns(args, "apply_bn_replace")
        self.reduce_range = get_ns(args, "reduce_range")
        self.include_encoder_stem = get_ns(args, "include_encoder_stem")
        self.include_decoder_stem = get_ns(args, "include_decoder_stem")

        self.param_type = get_ns(args, "param_type")
        self.layer_type = get_ns(args, "layer_type")
        self.block_type = get_ns(args, "block_type")
        self.norm_type = get_ns(args, "norm_type")

  def initialize_default_config(self) -> None:
    self.shared_preprocess = 0
    self.apply_zero_init = 0
    self.apply_bn_tracking = 1
    self.apply_bn_calibrate = 0
    self.apply_bn_replace = 1
    self.reduce_range = 1
    self.include_encoder_stem = 0
    self.include_decoder_stem = 0
    self.param_type = "post_act"
    self.layer_type = "sig_gate"
    self.block_type = "linear"
    self.norm_type = "scale_shift"
