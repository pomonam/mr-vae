import argparse
import os

import numpy as np
import torch
import tqdm
import wandb
import glob
from src.evaluate import AverageMeter
from experiments.mnist.input_pipeline import build_input_queue
from experiments.mnist.model_pipeline import build_criterion
from experiments.mnist.model_pipeline import build_model
from experiments.init_wandb import init_wandb
from src.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import tile_image
from src.utils import seed_everything
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hypervae-mnist-train")

parser.add_argument("--encoder_name", type=str, default="cnn")
parser.add_argument("--decoder_name", type=str, default="cnn")

parser.add_argument("--total_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def main():

    cfg = TrainConfig(args)

    seed_everything(cfg.seed)
    model = build_model(args.encoder_name, args.decoder_name, DEVICE)

    file_lst = glob.glob(os.path.join(args.checkpoint_dir, "base_*"))

    for f in file_lst:
        if "cyclic" in f:
            print(f)
            if DEVICE == torch.device("cpu"):
                model.load_state_dict(torch.load(f, map_location=torch.device("cpu"))["state_dict"])
            else:
                model.load_state_dict(torch.load(f)["state_dict"])

            model.eval()
            n = 3
            m = 5
            num_samples = n * m

            output_img_lst = []
            for _ in range(num_samples):
                output_img = model.prior_sample(DEVICE)
                output_img_lst.append(output_img)
            # print(output_img_lst)
            output_img = torch.concat(output_img_lst)
            output_img = 1. - torch.sigmoid(output_img)
            output_tiled = tile_image(output_img, n, m)
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.imshow(output_tiled.detach().cpu().permute(1, 2, 0).numpy(), cmap="Greys")
            plt.title(f)
            plt.show()

            # Visualizing the reconstruction
            # test_loader = build_input_queue("test", cfg.batch_size, DEVICE, data_path="../../../logs/data")
            # test_batch = next(test_loader)
            # outputs_dict = model.forward(test_batch["inputs"])
            # logits = outputs_dict["logits"].view(-1, 28, 28)
            # plt.figure(figsize=(5, 5))
            # plt.axis("square")
            # for i in range(50):
            #     data_i = test_batch["inputs"].view(-1, 28, 28)[i].data.cpu().numpy()
            #     recon_i = torch.sigmoid(logits[i]).data.cpu().numpy()
            #     plt.subplot(10, 10, 2 * i + 1)
            #     plt.imshow(data_i, cmap="Greys")
            #     plt.axis("off")
            #     plt.subplot(10, 10, 2 * i + 2)
            #     plt.imshow(recon_i, cmap="Greys")
            #     plt.axis("off")
            # plt.show()
            # break
        else:
            pass


    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # criterion = build_criterion(DEVICE)
    #
    # train(model, build_input_queue, criterion, optimizer, cfg)
    #
    # if args.save_eval_checkpoint is not None:
    #     save_checkpoint = os.path.join("checkpoints", "base_{}_{}.pth".format(args.beta,
    #                                                                           args.schedule))
    #     log_info = {
    #         "state_dict": model.state_dict(),
    #     }
    #     torch.save(log_info, save_checkpoint)
    #
    # evaluate(model,
    #          build_input_queue,
    #          criterion,
    #          cfg.total_epochs,
    #          "train_eval")
    # evaluate(model, build_input_queue, criterion, cfg.total_epochs, "test")


if __name__ == "__main__":
    main()
