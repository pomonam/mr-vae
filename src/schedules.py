import numpy as np


def build_betas_schedule(name: str, beta: float, epochs: int) -> np.ndarray:
  if name == "constant":
    beta_schedule = np.ones(epochs) * beta
  elif name == "monotonic":
    beta_schedule = frange_cycle_linear(0, beta, epochs, 1, 0.25)
  elif name == "cyclic":
    beta_schedule = frange_cycle_linear(0, beta, epochs, 4, 0.5)
  else:
    raise ValueError
  return beta_schedule


def frange_cycle_linear(start: float,
                        stop: float,
                        n_epoch: int,
                        n_cycle: int = 4,
                        ratio: float = 0.5) -> np.ndarray:
  total_beta = np.ones(n_epoch) * stop
  period = n_epoch / n_cycle
  step = (stop - start) / (period * ratio)

  for c in range(n_cycle):
    v, i = start, 0
    while v <= stop and (int(i + c * period) < n_epoch):
      total_beta[int(i + c * period)] = v
      v += step
      i += 1
  return total_beta


if __name__ == "__main__":
  res = frange_cycle_linear(0, 1, 200, n_cycle=1, ratio=0.25)
  print(res)
