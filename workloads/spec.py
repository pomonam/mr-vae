import abc
import enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import os
import torch
import copy


class Workload(metaclass=abc.ABCMeta):
  def save_checkpoint(self, dir_path, epoch: int):
    checkpoint_dir = os.path.join(dir_path, f"checkpoint_epoch_{epoch}")

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    # save optimizer
    torch.save(
      copy.deepcopy(self.optimizer.state_dict()),
      os.path.join(checkpoint_dir, "optimizer.pt"),
    )

    # save scheduler
    torch.save(
      copy.deepcopy(self.scheduler.state_dict()),
      os.path.join(checkpoint_dir, "scheduler.pt"),
    )

    # save model
    self.model.save(checkpoint_dir)

    # save training config
    self.train_cfg.save_json(checkpoint_dir, "training_config")

  def save_model(self, dir_path: str):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

    # save model
    self.model.save(dir_path)

    # save training config
    self.train_cfg.save_json(dir_path, "training_config")
