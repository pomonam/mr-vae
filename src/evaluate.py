"""Lightweight running-average metrics for training loops."""


class AverageMeter:

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = None
    self.sum = 0.0
    self.count = 0

  def update(self, val, n=1):
    if val is None or n <= 0:
      return
    self.val = val
    self.sum += val * n
    self.count += n

  @property
  def avg(self):
    return self.sum / self.count if self.count > 0 else self.val


def initialize_metric(metric_names):
  return {name: AverageMeter() for name in metric_names}


def update_metric(metric_dict, log_dict, n=1):
  for k, v in log_dict.items():
    metric_dict[k].update(v.item() if hasattr(v, "item") else v, n)
  return metric_dict


def summarize_metric(metric_dict, name=""):
  return {name + k: metric_dict[k].avg for k in metric_dict}


def generate_metric_str(name, epoch, summ_dict):
  body = "".join(
      "\t{} = {}".format(k, round(v, 4) if v is not None else None)
      for k, v in summ_dict.items())
  return "Epoch {:d}\t{}\t|{}".format(epoch, name, body)
