import numpy as np


def build_betas_schedule(name, beta, epochs):
    if name == "constant":
        beta_schedule = np.ones(epochs) * beta
    elif name == "monotonic":
        beta_schedule = frange_cycle_linear(0, beta, epochs, 1, 0.25)
    elif name == "cyclic":
        beta_schedule = frange_cycle_linear(0, beta, epochs, 4)
    else:
        raise ValueError("Invalid Schedule")
    return beta_schedule


def frange_cycle_linear(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch) * stop
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


if __name__ == "__main__":
    res = frange_cycle_linear(0, 1, 200, n_cycle=1, ratio=0.25)
    print(res)
