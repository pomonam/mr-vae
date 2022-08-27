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
    workload: str = None
    data_name: str = None
    arch_name: str = None

    output_dir: str = None
    batch_size: int = 128
    num_epochs: int = 100

    lr: float = 1e-4
    beta: float = 1e-4
    schedule: str = None
    checkpoint_dir: str = None

    seed: int = 8
