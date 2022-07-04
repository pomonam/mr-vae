import torch
import numpy as np


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
  L = np.ones(n_epoch)
  period = n_epoch / n_cycle
  # linear schedule
  step = (stop - start) / (period * ratio)

  for c in range(n_cycle):

    v, i = start, 0
    while v <= stop and (int(i + c * period) < n_epoch):
      L[int(i + c * period)] = v
      v += step
      i += 1
  return L

