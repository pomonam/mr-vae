from argparse import Namespace
from typing import Any, Optional

from src.schedules import build_betas_schedule


def get_ns(args: Namespace, name: str) -> Optional[Any]:
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

  def get_beta(self, epoch: int) -> Optional[float]:
    if self.beta_schedule is not None:
      return self.beta_schedule[epoch]
    return None
