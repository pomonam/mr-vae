# pylint: skip-file

import bisect
from collections import OrderedDict
import time
from typing import Dict, Optional

import numpy as np
import torch


def tile_image(batch_image, n, m=None):
  if m is None:
    m = n
  assert n * m == batch_image.size(0)
  channels, height, width = (
      batch_image.size(1),
      batch_image.size(2),
      batch_image.size(3),
  )
  batch_image = batch_image.view(n, m, channels, height, width)
  batch_image = batch_image.permute(2, 0, 4, 1, 3)
  batch_image = batch_image.contiguous().view(channels, n * height, m * width)
  return batch_image


def type_as(a, b):
  if torch.is_tensor(a) and torch.is_tensor(b):
    return a.to(b)
  else:
    return a


class Meter(object):

  def __init__(self):
    pass

  def state_dict(self):
    return {}

  def load_state_dict(self, state_dict):
    pass

  def reset(self):
    raise NotImplementedError

  @property
  def smoothed_value(self) -> float:
    """Smoothed value used for logging."""
    raise NotImplementedError


def safe_round(number, ndigits):
  if hasattr(number, "__round__"):
    return round(number, ndigits)
  elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
    return safe_round(number.item(), ndigits)
  elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
    return safe_round(number.item(), ndigits)
  else:
    return number


class AverageMeter(Meter):

  def __init__(self, round=None):
    self.round = round
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    if val is not None:
      self.val = val
      if n > 0:
        self.sum = type_as(self.sum, val) + (val * n)
        self.count = type_as(self.count, n) + n

  def state_dict(self):
    return {
        "val": self.val,
        "sum": self.sum,
        "count": self.count,
        "round": self.round,
    }

  def load_state_dict(self, state_dict):
    self.val = state_dict["val"]
    self.sum = state_dict["sum"]
    self.count = state_dict["count"]
    self.round = state_dict.get("round", None)

  @property
  def avg(self):
    return self.sum / self.count if self.count > 0 else self.val

  @property
  def smoothed_value(self) -> float:
    val = self.avg
    if self.round is not None and val is not None:
      val = safe_round(val, self.round)
    return val


class SumMeter(Meter):

  def __init__(self, round: Optional[int] = None):
    self.round = round
    self.reset()

  def reset(self):
    self.sum = 0

  def update(self, val):
    if val is not None:
      self.sum = type_as(self.sum, val) + val

  def state_dict(self):
    return {
        "sum": self.sum,
        "round": self.round,
    }

  def load_state_dict(self, state_dict):
    self.sum = state_dict["sum"]
    self.round = state_dict.get("round", None)

  @property
  def smoothed_value(self) -> float:
    val = self.sum
    if self.round is not None and val is not None:
      val = safe_round(val, self.round)
    return val


class TimeMeter(Meter):

  def __init__(
      self,
      init: int = 0,
      n: int = 0,
      round: Optional[int] = None,
  ):
    self.round = round
    self.reset(init, n)

  def reset(self, init=0, n=0):
    self.init = init
    self.start = time.perf_counter()
    self.n = n
    self.i = 0

  def update(self, val=1):
    self.n = type_as(self.n, val) + val
    self.i += 1

  def state_dict(self):
    return {
        "init": self.elapsed_time,
        "n": self.n,
        "round": self.round,
    }

  def load_state_dict(self, state_dict):
    if "start" in state_dict:
      # backwards compatibility for old state_dicts
      self.reset(init=state_dict["init"])
    else:
      self.reset(init=state_dict["init"], n=state_dict["n"])
      self.round = state_dict.get("round", None)

  @property
  def avg(self):
    return self.n / self.elapsed_time

  @property
  def elapsed_time(self):
    return self.init + (time.perf_counter() - self.start)

  @property
  def smoothed_value(self) -> float:
    val = self.avg
    if self.round is not None and val is not None:
      val = safe_round(val, self.round)
    return val


class StopwatchMeter(Meter):

  def __init__(self, round: Optional[int] = None):
    self.round = round
    self.sum = 0
    self.n = 0
    self.start_time = None

  def start(self):
    self.start_time = time.perf_counter()

  def stop(self, n=1, prehook=None):
    if self.start_time is not None:
      if prehook is not None:
        prehook()
      delta = time.perf_counter() - self.start_time
      self.sum = self.sum + delta
      self.n = type_as(self.n, n) + n

  def reset(self):
    self.sum = 0
    self.n = 0
    self.start()

  def state_dict(self):
    return {
        "sum": self.sum,
        "n": self.n,
        "round": self.round,
    }

  def load_state_dict(self, state_dict):
    self.sum = state_dict["sum"]
    self.n = state_dict["n"]
    self.start_time = None
    self.round = state_dict.get("round", None)

  @property
  def avg(self):
    return self.sum / self.n if self.n > 0 else self.sum

  @property
  def elapsed_time(self):
    if self.start_time is None:
      return 0.0
    return time.perf_counter() - self.start_time

  @property
  def smoothed_value(self) -> float:
    val = self.avg if self.sum > 0 else self.elapsed_time
    if self.round is not None and val is not None:
      val = safe_round(val, self.round)
    return val


class MetersDict(OrderedDict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.priorities = []

  def __setitem__(self, key, value):
    assert key not in self, "MetersDict doesn't support reassignment"
    priority, value = value
    bisect.insort(self.priorities, (priority, len(self.priorities), key))
    super().__setitem__(key, value)
    for _, _, key in self.priorities:  # reorder dict to match priorities
      self.move_to_end(key)

  def add_meter(self, key, meter, priority):
    self.__setitem__(key, (priority, meter))

  def state_dict(self):
    return [
        (pri, key, self[key].__class__.__name__, self[key].state_dict())
        for pri,
        _,
        key in self.priorities
        # can't serialize DerivedMeter instances
        if not isinstance(self[key], MetersDict._DerivedMeter)
    ]

  def load_state_dict(self, state_dict):
    self.clear()
    self.priorities.clear()
    for pri, key, meter_cls, meter_state in state_dict:
      meter = globals()[meter_cls]()
      meter.load_state_dict(meter_state)
      self.add_meter(key, meter, pri)

  def get_smoothed_value(self, key: str) -> float:
    meter = self[key]
    if isinstance(meter, MetersDict._DerivedMeter):
      return meter.fn(self)
    else:
      return meter.smoothed_value

  def get_smoothed_values(self) -> Dict[str, float]:
    return OrderedDict([(key, self.get_smoothed_value(key))
                        for key in self.keys()
                        if not key.startswith("_")])

  def reset(self):
    for meter in self.values():
      if isinstance(meter, MetersDict._DerivedMeter):
        continue
      meter.reset()

  class _DerivedMeter(Meter):

    def __init__(self, fn):
      self.fn = fn

    def reset(self):
      pass


def initialize_metric(metric_lst):
  metric_dict = {}
  for m in metric_lst:
    metric_dict[m] = AverageMeter()
  return metric_dict


def update_metric(metric_dict, log_dict, n=1):
  for m in log_dict.keys():
    metric_dict[m].update(log_dict[m].item(), n)
  return metric_dict


def summarize_metric(metric_dict, name=""):
  summ_dict = {}
  for m in metric_dict.keys():
    summ_dict[name + str(m)] = metric_dict[m].avg
  return summ_dict


def generate_metric_str(name, epoch, summ_dict):
  summ_str = ""
  for key in summ_dict.keys():
    summ_str += "\t{} = {}".format(key, round(summ_dict[key], 4))
  return "Epoch {:d}\t{}\t|{}".format(epoch, name, summ_str)


def mean(res, key):
  return torch.stack(
      [x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()
