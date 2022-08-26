import json
import os
import warnings
from dataclasses import asdict, field
from typing import Any, Dict, Union

from pydantic import ValidationError
from pydantic.dataclasses import dataclass
from pythae.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """
    BaseTrainer config class stating the main training arguments.
    Parameters:
        output_dir (str): The directory where model checkpoints, configs and final
            model will be stored. Default: None.
        batch_size (int): The number of training samples per batch. Default 100
        num_epochs (int): The maximal number of epochs for training. Default: 100
        learning_rate (int): The learning rate applied to the `Optimizer`. Default: 1e-4
        steps_saving (int): A model checkpoint will be saved every `steps_saving` epoch.
            Default: None
        steps_saving (int): A prediction using the best model will be run every `steps_predict`
            epoch. Default: None
        keep_best_on_train (bool): Whether to keep the best model on the train set. Default: False.
        seed (int): The random seed for reproducibility
        no_cuda (bool): Disable `cuda` training. Default: False
    """
    workload: str = None
    data_name: str = None
    arch_name: str = None

    output_dir: str = None
    batch_size: int = 100
    num_epochs: int = 100
    lr: float = 1e-4
    beta: float = 1e-4
    schedule: str = None
    checkpoint_dir: str = None

    steps_saving: Union[int, None] = None
    steps_predict: Union[int, None] = None
    keep_best_on_train: bool = False
    seed: int = 8
    no_cuda: bool = False
