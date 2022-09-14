import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.wandb_utils import init_api


def init_plotting():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.aistats2022(column="full"))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())
