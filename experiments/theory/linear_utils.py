from math import pi

import torch

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
    W = model.decoder.layer.state_dict()["weight"]

    WT = torch.t(W).to(DEVICE)
    I_x = torch.eye(data.size()[-1]).to(DEVICE)
    I_z = torch.eye(W.size()[-1]).to(DEVICE)
    data = torch.t(data).to(DEVICE)
    U, D, V = torch.svd(W.to(DEVICE))
    denominator = 1. / beta + D**2
    diagonal = torch.diag(torch.div(D, denominator)).to(DEVICE)
    core = torch.matmul(torch.matmul(V, diagonal), torch.t(U)).to(DEVICE)
    mu = torch.matmul(core.to(DEVICE), (data))
    cov = I_z - torch.matmul(core, W.to(DEVICE))

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
    log_const = torch.tensor(0) # (latent_size / 2.) * torch.log(torch.tensor(2. * pi))
    W = model.decoder.layer.state_dict()["weight"]
    WT = torch.t(W)
    xb_dot_product = torch.sum(torch.mul((data), (data)), dim=1)
    xb_dot_batch_mean = torch.mean(xb_dot_product)
    cross_term_batch = torch.sum(
        torch.mul(torch.t(torch.matmul(W, mu)), (data)), dim=1)
    cross_term_batch_mean = torch.mean(cross_term_batch)
    E_Y = torch.matmul(W, mu)
    cov_Y = torch.matmul(torch.matmul(W, cov), WT)
    E_Y_squared_batch = torch.sum(torch.mul(E_Y, E_Y), dim=0)
    E_Y_squared_batch_mean = torch.mean(E_Y_squared_batch)
    E_Wz = E_Y_squared_batch_mean + torch.trace(cov_Y)
    distortion = log_const + (
        0.5 * xb_dot_batch_mean) - cross_term_batch_mean + (0.5 * E_Wz)

    return distortion


def analytic_rate_and_distortion(model, rd_data, beta):
    rate_list = list()
    distortion_list = list()
    for data in rd_data:
        inputs = data['inputs'].reshape(data['inputs'].size()[0], -1)
        q_mean, q_cov, D = analytical_q_svd(inputs, model, beta)

        rate = analytical_rate_point(inputs, q_mean, q_cov, beta, D).item()
        distortion = analytical_distortion_point(inputs, q_mean, q_cov,
                                                 model).item()

        rate_list.append(rate)
        distortion_list.append(distortion)

    rate_expecation = sum(rate_list) / len(rate_list)
    distortion_expectation = sum(distortion_list) / len(distortion_list)

    return rate_expecation, distortion_expectation
