import os
import random
from typing import Optional

import numpy as np
import torch


def log_sum_exp(value: torch.Tensor,
                dim: Optional[int] = None,
                keepdim: bool = False) -> torch.Tensor:
  if dim is not None:
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
      m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
  else:
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


def _select_seed_randomly(min_seed_value: int = 0,
                          max_seed_value: int = 255) -> int:
  return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: int) -> int:
  max_seed_value = np.iinfo(np.uint32).max
  min_seed_value = np.iinfo(np.uint32).min

  try:
    if seed is None:
      seed = os.environ.get("PL_GLOBAL_SEED")
    seed = int(seed)
  except (TypeError, ValueError):
    seed = _select_seed_randomly(min_seed_value, max_seed_value)
    print(f"No correct seed found, seed set to {seed}")

  if not min_seed_value <= seed <= max_seed_value:
    print(
        f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
    )
    seed = _select_seed_randomly(min_seed_value, max_seed_value)

  os.environ["PL_GLOBAL_SEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  return seed
