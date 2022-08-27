import argparse
import os

import numpy as np
import torch
import tqdm
import wandb
# from absl import logging
from src.evaluate import AverageMeter
# from experiments.mnist.input_pipeline import build_input_queue
# from experiments.mnist.model_pipeline import build_criterion
# from experiments.mnist.model_pipeline import build_model
# from experiments.init_wandb import init_wandb
from workloads.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.utils import seed_everything
from absl import app
from absl import flags
from absl import logging
from workloads.binary_image.workload import BinaryImageWorkload
import logging
import datetime

flags.DEFINE_string('workload', 'binary_image', 'Name of the workload')
flags.DEFINE_string('data_name', 'mnist', 'Name of the workload')
flags.DEFINE_string('arch_name', 'constant', 'Name of the workload')
flags.DEFINE_string('schedule', 'constant', 'Name of the workload')
flags.DEFINE_float('beta', 1, 'Name of the workload')

flags.DEFINE_integer('num_epochs', 1000, 'Name of the workload')
flags.DEFINE_integer('seed', 0, 'Name of the workload')
flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Name of the workload')


FLAGS = flags.FLAGS

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def main(_):
  train_dict = {
    "workload": FLAGS.workload,
    "data_name": FLAGS.data_name,
    "arch_name": FLAGS.arch_name,
    "num_epochs": FLAGS.num_epochs,
    "beta": FLAGS.beta,
    "schedule": FLAGS.schedule,
    "seed": FLAGS.seed,
    "checkpoint_dir": FLAGS.checkpoint_dir,
  }
  train_cfg = TrainConfig().from_dict(train_dict)
  workload = BinaryImageWorkload(train_cfg, FLAGS.data_name, FLAGS.arch_name)

  logging.info("Training started !\n")
  logging.info(
    f"Training params:\n - max_epochs: {FLAGS.num_epochs}\n"
    f" - batch_size: {workload.batch_size}\n"
    f" - checkpoint saving every {workload.save_interval}\n"
  )

  logging.info(f"Model Architecture: {workload.model}\n")
  logging.info(f"Optimizer: {workload.optimizer}\n")

  for epoch in range(1, FLAGS.num_epochs + 1):

    workload.callback_handler.on_epoch_begin(
      training_config=train_cfg,
      epoch=epoch,
      train_loader=workload.train_loader,
      eval_loader=workload.eval_loader,
    )

    metrics = {}

    epoch_train_loss = workload.train_step(epoch)
    metrics["train_epoch_loss"] = epoch_train_loss

    epoch_eval_loss = workload.eval_step(epoch)
    metrics["eval_epoch_loss"] = epoch_eval_loss
    workload.scheduler_step(epoch_eval_loss)

    if epoch % workload.predict_interval == 0:
      true_data, reconstructions, generations = workload.predict()

      workload.callback_handler.on_prediction_step(
        train_cfg,
        true_data=true_data,
        reconstructions=reconstructions,
        generations=generations,
        global_step=epoch,
      )

    workload.callback_handler.on_epoch_end(training_config=workload.train_cfg)

    # save checkpoints
    if epoch % workload.save_interval == 0:
      workload.save_checkpoint(dir_path=FLAGS.checkpoint_dir, epoch=epoch)
      logging.info(f"Saved checkpoint at epoch {epoch}\n")

    workload.callback_handler.on_log(
      train_cfg, metrics, logger=logging, global_step=epoch
    )

  final_dir = os.path.join(FLAGS.checkpoint_dir, "final_model")
  workload.save_model(dir_path=final_dir)
  logging.info("Training ended!")
  logging.info(f"Saved final model in {final_dir}")

  workload.callback_handler.on_train_end(workload.train_cfg)


if __name__ == "__main__":
  app.run(main)
