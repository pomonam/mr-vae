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
from workloads.binary_image.workload import BinaryImageWorkload
import logging
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--workload", type=str, default="binary_image")

parser.add_argument("--data_name", type=str, default="mnist")
parser.add_argument("--arch_name", type=str, default="constant")

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)

args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

logger = logging.getLogger(__name__)


def main():
  train_cfg = TrainConfig()
  train_cfg.from_dict(vars(args))
  workload = BinaryImageWorkload(train_cfg, args.data_name, args.arch_name)

  logger.info("Model passed sanity check !\n")

  training_signature = (
    str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
  )

  # training_dir = os.path.join(
  #   self.training_config.output_dir,
  #   f"{self.model.model_name}_training_{self._training_signature}",
  # )
  #
  # self.training_dir = training_dir
  training_dir = "."
  log_output_dir = "."
  if not os.path.exists(training_dir):
    os.makedirs(training_dir)
    logger.info(
      f"Created {training_dir}. \n"
      "Training config, checkpoints and final model will be saved here.\n"
    )

  log_verbose = False

  # set up log file
  if log_output_dir is not None:
    log_dir = log_output_dir
    log_verbose = True

    # if dir does not exist create it
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
      logger.info(f"Created {log_dir} folder since did not exists.")
      logger.info("Training logs will be recodered here.\n")
      logger.info(" -> Training can be monitored here.\n")

    # create and set logger
    log_name = f"training_logs_{training_signature}"

    file_logger = logging.getLogger(log_name)
    file_logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(
      os.path.join(log_dir, f"training_logs_{training_signature}.log")
    )
    f_handler.setLevel(logging.INFO)
    file_logger.addHandler(f_handler)

    # Do not output logs in the console
    file_logger.propagate = False

    file_logger.info("Training started !\n")
    file_logger.info(
      f"Training params:\n - max_epochs: {workload.num_epochs}\n"
      f" - batch_size: {workload.batch_size}\n"
      f" - checkpoint saving every {workload.save_interval}\n"
    )

    file_logger.info(f"Model Architecture: {workload.model}\n")
    file_logger.info(f"Optimizer: {workload.optimizer}\n")

  logger.info("Successfully launched training !\n")

  for epoch in range(1, workload.num_epochs + 1):

    workload.callback_handler.on_epoch_begin(
      training_config=workload.train_cfg,
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

    if epoch % workload.predict_inteval == 0:
      true_data, reconstructions, generations = workload.predict()

      workload.callback_handler.on_prediction_step(
        workload.train_cfg,
        true_data=true_data,
        reconstructions=reconstructions,
        generations=generations,
        global_step=epoch,
      )

    workload.callback_handler.on_epoch_end(training_config=workload.train_cfg)

    # save checkpoints
    if epoch % workload.save_interval == 0:
      workload.save_checkpoint(dir_path=training_dir, epoch=epoch)
      logger.info(f"Saved checkpoint at epoch {epoch}\n")

    workload.callback_handler.on_log(
      workload.train_cfg, metrics, logger=logger, global_step=epoch
    )

  final_dir = os.path.join(training_dir, "final_model")
  workload.save_model(dir_path=final_dir)
  logger.info("Training ended!")
  logger.info(f"Saved final model in {final_dir}")

  workload.callback_handler.on_train_end(workload.train_cfg)


if __name__ == "__main__":
  main()
