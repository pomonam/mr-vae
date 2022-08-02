import torch
from torch import nn
import argparse
import numpy as np
import wandb
import tqdm
import os
os.environ["WANDB_API_KEY"] = "65a71cb86f66a117460fb632080693d4cc9ab979"
from math import pi
import matplotlib.pyplot as plt

from experiments.b_mnist.input_pipeline import build_input_queue
from experiments.b_mnist.model_pipeline import build_criterion
from experiments.b_mnist.model_pipeline import build_hyper_model
from experiments.b_mnist.results.rd_curve import get_rd
from experiments.init_wandb import init_api
from experiments.init_wandb import init_wandb
from src.config import HyperConfig
from src.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.plotting import init_plotting
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hyper_vae-hyper-b_mnist_linear")

parser.add_argument("--training_method", type=str, default="sequential",
                    choices=["simultaneous", "sequential"])
parser.add_argument("--hyper_type", type=str, default="add")
parser.add_argument("--block_type", type=str, default="linear")
parser.add_argument("--include_output_layer", type=int, default=1)
parser.add_argument("--include_sigmoid_activation", type=int, default=1)
parser.add_argument("--preprocess_beta", type=int, default=0)
parser.add_argument("--sample_type", type=str, default="fixed_log_uniform")
parser.add_argument("--sample_range", type=tuple, default=(1e-3, 10))

parser.add_argument("--total_epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=500)

args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# =====================================================================================================================
# Analytical Rate & Distortion Utility Functions Adapted From
# GitHub: https://github.com/BorealisAI/rate_distortion/blob/master/rate_distortion/algorithms/analytical_linear_vae.py
# =====================================================================================================================

def analytical_q_svd(data, model, beta=1):
    '''
    Compute the mean and covariance of the analytical q_beta with SVD

    '''
    W = model.decoder.linear1.state_dict()["module.weight"].cpu()

    WT = torch.t(W)
    I_x = torch.eye(data.size()[-1])
    I_z = torch.eye(W.size()[-1])
    data = torch.t(data)
    U, D, V = torch.svd(W)
    denominator = 1. / beta + D**2
    diagonal = torch.diag(torch.div(D, denominator))
    core = torch.matmul(torch.matmul(V, diagonal), torch.t(U))
    mu = torch.matmul(core, (data))
    cov = I_z - torch.matmul(core, W)

    return mu, cov, D


def analytical_rate_point(data, mu, cov, beta, singular_values):
    '''
    compute the optimal rate for a batch of data
    '''

    latent_size = int(list(cov.size())[0])

    if singular_values is not None:
        first_term = torch.log(torch.tensor(
            1. / beta)) * singular_values.size()[0]
        second_term = torch.sum(torch.log(singular_values**2 + (1. / beta)))
        log_det = first_term - second_term
        trace_cov = torch.sum((1. / beta) / (singular_values**2 + (1. / beta)))

    else:
        det_cov = torch.det(cov)
        log_det = torch.log(det_cov)
        trace_cov = torch.trace(cov)

    mu_product = torch.sum(torch.mul(mu, mu), dim=0)
    mu_batch_mean = torch.mean(mu_product)
    rate = 0.5 * (trace_cov + mu_batch_mean - latent_size - log_det)
    return rate


def analytical_distortion_point(data, mu, cov, model):
    '''
    compute the optimal distortion for a batch of data
    '''
    latent_size = int(list(data.size())[-1])
    log_const = (latent_size / 2.) * torch.log(torch.tensor(2. * pi))
    W = model.decoder.linear1.state_dict()["module.weight"].cpu()
    WT = torch.t(W)
    xb_dot_product = torch.sum(torch.mul((torch.transpose(data, dim0=0, dim1=1)), (data)), dim=1)
    xb_dot_batch_mean = torch.mean(xb_dot_product)
    cross_term_batch = torch.sum(
        torch.mul(torch.t(torch.matmul(W, mu)), (data)), dim=1)
    cross_term_batch_mean = torch.mean(cross_term_batch)
    E_Y = torch.matmul(W, mu)
    cov_Y = torch.matmul(torch.matmul(W, cov), WT)
    E_Y_squared_batch = torch.sum(torch.mul(E_Y, E_Y), dim=0)
    E_Y_squared_batch_mean = torch.mean(E_Y_squared_batch)
    E_Wz = E_Y_squared_batch_mean + torch.trace(cov_Y)
    distortion = log_const + 0.5 * xb_dot_batch_mean - cross_term_batch_mean + 0.5 * E_Wz

    return distortion


def analytic_rate_and_distortion(model, rd_data, beta):
    rate_list = list()
    distortion_list = list()
    for data in rd_data:
        inputs = data['inputs'].reshape(data['inputs'].size()[0], -1).cpu()
        q_mean, q_cov, D = analytical_q_svd(inputs, model, beta)

        rate = analytical_rate_point(inputs, q_mean, q_cov, beta, D).item()
        distortion = analytical_distortion_point(inputs, q_mean, q_cov, model).item()

        rate_list.append(rate)
        distortion_list.append(distortion)
    
    rate_expecation = sum(rate_list)/len(rate_list)
    distortion_expectation = sum(distortion_list)/len(distortion_list)

    return rate_expecation, distortion_expectation

# ================================================================================================
# Linear VAE Class Adapted From
# GitHub: https://github.com/BorealisAI/rate_distortion/blob/master/rate_distortion/models/vaes.py
# ================================================================================================

def log_normal_likelihood(x, mean, logvar):
    """Implementation WITH constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    Args:
        x: [B,Z]
        mean,logvar: [B,Z]
    Returns:
        output: [B]
    """

    dim = list(mean.size())[1]
    logvar = (torch.zeros(mean.size()) + logvar).to(DEVICE)
    return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(1) +
                   torch.log(torch.tensor(2 * pi)) * dim)


def log_mean_exp(x, dim=1):
    """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
    max_, _ = torch.max(x, dim, keepdim=True, out=None)
    return torch.log(torch.mean(torch.exp(x - max_), dim)) + torch.squeeze(max_)


def log_normal(x, mean, logvar):
    """
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    log normal WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out later
    Args:s
        x: [B,Z]
        mean,logvar: [B,Z]
    Returns:
        output: [B]
    """
    return -0.5 * (logvar.sum(1) + (
        (x - mean).pow(2) / torch.exp(logvar)).sum(1))


def singleton_repeat(x, n):
    """ 
    based on https://github.com/BorealisAI/rate_distortion/blob/master/rate_distortion/utils/computation_utils.py
    Repeat a batch of data n times. 
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back. 
    So that later when reshaping, it's guranteed to follow the same tensor convention. 
     """
    if n == 1:
        return x
    else:
        singleton_x = torch.unsqueeze(x, 0)
        repeated_x = singleton_x.repeat(n, 1, 1)
        return repeated_x.view(-1, x.size()[-1])


class VAE(nn.Module):

    def __init__(self, input_shape, bottleneck_size):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.input_num_elements = int(torch.prod(torch.Tensor(input_shape)).item())
        self.bottleneck_size = bottleneck_size
        self.enc = nn.Linear(self.input_num_elements, bottleneck_size*2, bias=False)
        self.dec = nn.Linear(bottleneck_size, self.input_num_elements, bias=False)
        self.observation_log_likelihood_fn = log_normal_likelihood
        self.x_logvar = nn.Parameter(torch.log(torch.tensor(1)), requires_grad=True)

    def encode(self, x):
        hidden = self.enc(x)
        mean = hidden[:, :self.bottleneck_size]
        logvar = hidden[:, self.bottleneck_size:]
        return mean, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        logqz = log_normal(z, mu, logvar)
        zeros = torch.zeros_like(z)
        logpz = log_normal(z, zeros, zeros)
        return z, logpz, logqz

    def decode(self, z):
        return self.dec(z), torch.zeros(1)

    def forward(self, x, num_iwae=1):
        flattened_x = x.view(-1, self.input_num_elements)
        flattened_x_k = singleton_repeat(flattened_x, num_iwae)
        mu, logvar = self.encode(flattened_x_k)
        z, logpz, logqz = self.reparameterize(mu, logvar)
        x_mean, x_logvar = self.decode(z)

        likelihood = self.observation_log_likelihood_fn(
            flattened_x_k, x_mean, x_logvar)
        elbo = likelihood + logpz - logqz

        if num_iwae != 1:
            elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
            logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
            logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
            likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
        elbo = torch.mean(elbo)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)
        likelihood = torch.mean(likelihood)
        
        output_dict = {
            "inputs": x,
            "logits": x_mean.reshape(tuple((-1,))+tuple(self.input_shape)),
            "mean": mu,
            "var": logvar
        }
        return elbo, output_dict

# ========
# Plotting
# ========

ENTITY = "lrscheduler114"
EXPERIMENT_NAME = "hv-b_mnist_hyper_mlp_linear"
ID = "k9m4y9yx"

def get_summary(summary):
    beta_to_rate = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/rate_lst"]))
    beta_to_dist = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/dist_lst"]))
    beta_to_elbo = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/loss_lst"]))
    
    beta_to_rate_analytical = dict(zip(summary["analytical/beta_lst"], summary["analytical/rate_lst"]))
    beta_to_dist_analytical = dict(zip(summary["analytical/beta_lst"], summary["analytical/dist_lst"]))
    return beta_to_rate, beta_to_dist, beta_to_elbo, beta_to_rate_analytical, beta_to_dist_analytical


def plot_rd_curves():
    init_plotting()

    api = init_api()
    runs = api.runs(ENTITY + "/" + EXPERIMENT_NAME)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished" and run.id == ID:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')})
            name_list.append(run.name)

    results = get_summary(summary_list[0])
    rate_dict, dist_dict, elbo_dict = results[0], results[1], results[2]
    analytical_rate_dict, analytical_dist_dict = results[3], results[4]

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    plt.plot(rate, dist, label="Hypernetwork")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")
    plt.scatter(rate, dist)


    keys = analytical_rate_dict.keys()
    values = zip(analytical_rate_dict.values(), analytical_dist_dict.values())
    analytical_combined_dict = dict(zip(keys, values))

    analytical_rate = np.array([c[0] for c in analytical_combined_dict.values()])
    analytical_dist = np.array([c[1] for c in analytical_combined_dict.values()])
    plt.plot(analytical_rate, analytical_dist, label='Analytical')

    min_val = min(np.min(rate), np.min(dist), np.min(analytical_rate), np.min(analytical_dist)) - 10
    max_val = min(np.max(rate), np.max(dist), np.max(analytical_rate), np.max(analytical_dist)) + 10
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")

    rate, dist = get_rd("hv-b_mnist_mlp_train-v5")
    plt.plot(rate, dist, label="Baseline")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    plt.legend()
    plt.show()


# =====================
# Hypernetwork Training
# =====================

def hyper_evaluate(model, criterion, epoch, name):
    model.eval()

    with torch.no_grad():
        beta_lst = np.logspace(-3, 1, num=20)
        loss_lst = []
        rate_lst = []
        dist_lst = []


        if name == "analytical":
            for beta in beta_lst:
                loader = build_input_queue(name, args.batch_size, DEVICE)
                rate, dist = analytic_rate_and_distortion(model, loader, beta)
                rate_lst.append(rate)
                dist_lst.append(dist)

                # HACK: idk how to do loss_lst rn
                loss_lst.append((beta*rate) + dist)
        else:
            for beta in beta_lst:
                metric_dict = initialize_metric(criterion.get_metric_lst())
                loader = build_input_queue(name, args.batch_size, DEVICE)
                p_bar = tqdm.tqdm(loader)

                for batch in p_bar:
                    inputs = batch["inputs"]
                    output_dict = model.fixed_forward(inputs, beta)
                    # We want to compute exact ELBO here
                    _, loss_dict = criterion.eval_forward(output_dict)

                    metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
                    summ_dict = summarize_metric(metric_dict)
                    summ_str = generate_metric_str(name, epoch, summ_dict)
                    p_bar.set_description(summ_str)

                summ_dict = summarize_metric(metric_dict, name="")
                loss_lst.append(summ_dict["loss"])
                rate_lst.append(summ_dict["rate"])
                dist_lst.append(summ_dict["distortion"])

        wandb.log({
            f"{name}/loss_lst": loss_lst,
            f"{name}/rate_lst": rate_lst,
            f"{name}/dist_lst": dist_lst,
            f"{name}/beta_lst": beta_lst
        })

        rd_data = [[x, y] for (x, y) in zip(rate_lst, dist_lst)]
        table = wandb.Table(data=rd_data, columns=["rate", "distortion"])
        wandb.log({f"{name}/rd_curve":
                   wandb.plot.line(table, "rate", "distortion", title="RD Curve")})

        loss_lst = np.array(loss_lst)
        rate_lst = np.array(rate_lst)
        dist_lst = np.array(dist_lst)

        total_auc = np.sum(loss_lst)
        top_auc = np.sum(loss_lst[:5])
        bot_auc = np.sum(loss_lst[5:])
        auc_dict = {
            f"{name}/total_auc": total_auc,
            f"{name}/top_auc": top_auc,
            f"{name}/bot_auc": bot_auc,
            f"{name}/max_rate": np.max(rate_lst),
            f"{name}/min_rate": np.min(rate_lst),
            f"{name}/abs_rate": np.max(rate_lst) - np.min(rate_lst),
            f"{name}/max_dist": np.max(dist_lst),
            f"{name}/min_dist": np.min(dist_lst),
            f"{name}/abs_dist": np.max(dist_lst) - np.min(dist_lst),

        }
        wandb.log(auc_dict)


def hyper_train(model, biq, criterion, optimizer, cfg, hyper_cfg):
    do_checkpoint = cfg.checkpoint_dir is not None
    if do_checkpoint and os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < cfg.total_epochs:
        do_evaluate = epoch % cfg.eval_freq == 0 and epoch != 0
        do_save = epoch % cfg.save_freq == 0 and epoch != 0

        if do_evaluate:
            hyper_evaluate(model, criterion, epoch, "train_eval")
            hyper_evaluate(model, criterion, epoch, "test")

        if do_checkpoint and do_save:
            slurm_check_dir = os.path.join(args.checkpoint_dir, "checkpoint.pth")
            log_info = {
                "id": wandb.run.id,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(log_info, slurm_check_dir)

        model.train()
        loader = biq("train", cfg.batch_size, DEVICE)
        p_bar = tqdm.tqdm(loader)
        metric_dict = initialize_metric(criterion.get_metric_lst())

        for batch in p_bar:
            inputs = batch["inputs"]

            output_dict = model(inputs)
            loss, loss_dict = criterion(output_dict, output_dict["beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
            summ_dict = summarize_metric(metric_dict)
            summ_str = generate_metric_str("train", epoch, summ_dict)
            p_bar.set_description(summ_str)

        summ_dict = summarize_metric(metric_dict, name="train_step/")
        wandb.log(summ_dict)
        epoch = epoch + 1

        if np.isnan(summ_dict["train_step/loss"]):
            wandb.finish(exit_code=1)
            raise ValueError()


def main():
    init_wandb(args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
    cfg = TrainConfig(args)
    hyper_cfg = HyperConfig(args)

    seed_everything(cfg.seed)
    model = VAE(input_shape=(28,28), bottleneck_size=64).to(DEVICE)
    model = build_hyper_model(args.encoder_name, args.decoder_name, hyper_cfg, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_criterion(DEVICE)
    hyper_train(model, build_input_queue, criterion, optimizer, cfg, hyper_cfg)
    hyper_evaluate(model, criterion, cfg.total_epochs, "analytical")
    hyper_evaluate(model, criterion, cfg.total_epochs, "train_eval")
    hyper_evaluate(model, criterion, cfg.total_epochs, "test")


if __name__ == "__main__":
    main()
    plot_rd_curves()
