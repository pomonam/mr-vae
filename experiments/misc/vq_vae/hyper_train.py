import argparse
import os

import numpy as np
import torch
import wandb

from experiments.image.input_pipeline import load_data
from experiments.misc.vq_vae.celeb_models import HyperVQCelebResNetDecoder
from experiments.misc.vq_vae.celeb_models import HyperVQCelebResNetEncoder
from experiments.misc.vq_vae.input_pipeline import load_mnist_data
from experiments.misc.vq_vae.mnist_models import HyperVQMNISTResNetDecoder
from experiments.misc.vq_vae.mnist_models import HyperVQMNISTResNetEncoder
from experiments.misc.vq_vae.hyper_train_utils import hyper_train
from experiments.misc.vq_vae.hyper_train_utils import hyper_predict
from experiments.misc.vq_vae.hyper_train_utils import hyper_evaluate
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.hyper.vq_vae import HyperVQVAE
from src.utils import seed_everything
from src.config import HyperConfig

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def build_model(data_name, hyper_cfg, device):
  if data_name == "mnist":
    model_config = {
        "commitment_loss_factor": 0.25,
        "quantization_loss_factor": 1.00,
        "num_embeddings": 256,
        "use_ema": False,
        "decay": 0.99,
        "input_dim": (1, 28, 28)
    }
    model = HyperVQVAE(
        model_config,
        encoder=HyperVQMNISTResNetEncoder(hyper_cfg),
        decoder=HyperVQMNISTResNetDecoder(hyper_cfg),
        hyper_cfg=hyper_cfg)
  else:
    model_config = {
        "commitment_loss_factor": 0.25,
        "quantization_loss_factor": 1.00,
        "num_embeddings": 1024,
        "use_ema": False,
        "decay": 0.99,
        "input_dim": (3, 64, 64)
    }
    model = HyperVQVAE(
        model_config,
        encoder=HyperVQCelebResNetEncoder(hyper_cfg),
        decoder=HyperVQCelebResNetDecoder(hyper_cfg),
        hyper_cfg=hyper_cfg)

  return model.to(device)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_name", type=str, default="hvq_image_debug")

  parser.add_argument("--data_name", type=str, default="mnist")

  parser.add_argument("--hyper_config_summary", type=str, default="lin_bn")

  parser.add_argument("--total_epochs", type=int, default=10)
  parser.add_argument("--warmup_epochs", type=int, default=10)

  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--batch_size", type=int, default=128)

  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--checkpoint_dir", type=str, default=None)
  parser.add_argument("--save_final_checkpoint", type=int, default=0)
  parser.add_argument("--save_freq", type=int, default=50)
  parser.add_argument("--eval_freq", type=int, default=10)
  args = parser.parse_args()

  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
  cfg = TrainConfig(args)
  hyper_cfg = HyperConfig(args)

  seed_everything(cfg.seed)
  model = build_model(args.data_name, hyper_cfg, DEVICE)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
  scheduler1 = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=1e-10,
      end_factor=1.,
      total_iters=cfg.warmup_epochs)
  cosine_epochs = max(cfg.total_epochs - cfg.warmup_epochs, 1)
  scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=cosine_epochs, eta_min=1e-6)
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[scheduler1, scheduler2],
      milestones=[cfg.warmup_epochs])

  if args.data_name == "mnist":
    train_loader = load_mnist_data(
        "train", cfg.batch_size, workers=4, data_path="../../../logs/data")
    test_loader = load_mnist_data(
        "test", cfg.batch_size, workers=2, data_path="../../../logs/data")
  else:
    train_loader = load_data(
        "celeba",
        "train",
        cfg.batch_size,
        workers=4,
        data_path="../../../logs/data")
    test_loader = load_data(
        "celeba",
        "test",
        cfg.batch_size,
        workers=4,
        data_path="../../../logs/data")

  hyper_train(
      model,
      train_loader,
      test_loader,
      optimizer,
      scheduler,
      DEVICE,
      cfg,
  )
  hyper_evaluate(model, train_loader, cfg.total_epochs, "train_eval", DEVICE)
  hyper_evaluate(model, test_loader, cfg.total_epochs, "test", DEVICE)

  for sample in model.get_test_samples(4):
    true_data, reconstructions, generations = hyper_predict(model, test_loader, sample, DEVICE)
    column_names = ["images_id", "truth", "reconstruction", "normal_generation"]
    data_to_log = []
    for i in range(len(true_data)):
      data_to_log.append([
          f"img_{i}",
          wandb.Image(np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)),
          wandb.Image(
              np.clip(
                  np.moveaxis(reconstructions[i].cpu().detach().numpy(), 0, -1),
                  0,
                  255.0,
              )),
          wandb.Image(
              np.clip(
                  np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1),
                  0,
                  255.0,
              )),
      ])
    val_table = wandb.Table(data=data_to_log, columns=column_names)
    wandb.log({"image_at_{}".format(sample): val_table})

  if args.save_final_checkpoint:
    save_checkpoint = \
      os.path.join("checkpoints", "base_{}_{}.pth".format(args.data_name, args.lamb))
    log_info = {
        "state_dict": model.state_dict(),
    }
    torch.save(log_info, save_checkpoint)

  wandb.finish()


if __name__ == "__main__":
  main()
