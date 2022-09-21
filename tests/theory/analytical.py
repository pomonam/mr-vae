import torch
import math

def analytical_rate(x, V, D):
    t1 = -torch.log(torch.det(D)).item()
    t2 = torch.mean(x.T @ V.T @ V @ x).item()
    t3 = torch.trace(D).item()
    t4 = -V.shape[0] #-(d/2)*torch.log(torch.tensor(2*math.pi)).item()
    return 0.5*(t1+t2+t3+t4)

def analytical_distortion(x, W, V, D):
    d = W.shape[0]
    '''
    t1 = -torch.trace(W @ D @ W.T).item()
    t2 = -torch.mean(x.T @ V.T @ W.T @ W @ V @ x).item()
    t3 = torch.mean(2*x.T @ W @ V @ x).item()
    t4 = -torch.mean(x.T @ x).item()
    t5 = -(d*torch.log(torch.tensor(2*math.pi))).item()
    return -0.5*(t1+t2+t3+t4+t5)
    '''
    t1 = -0.5*torch.mul(((W@V@x) - x).T, ((W@V@x) - x)).item()
    t2 = -(d*torch.log(torch.tensor(2*math.pi))).item()
    t3 = -torch.trace(W @ D @ W.T).item()
    return t1+t2+t3

def analytic_rate_and_distortion(model, rd_data, beta, DEVICE):
    rate_list = list()
    distortion_list = list()
    W = model.decoder.layer.state_dict()["weight"]
    beta = 1/beta #SOLELY TO INVERT THE OTHER BETA FLIP

    for data in rd_data:
        x = data['inputs'].reshape(data['inputs'].size()[0], -1).T

        # Calculate optimal encoder weights (V) and variance (D) given some decoder weight (W)
        V = torch.inverse((W.T @ W) + beta*torch.eye(W.shape[1], device=DEVICE)) @ W.T
        D = beta*torch.inverse(torch.diag(torch.diag(W.T @ W)) + beta*torch.eye(W.shape[1], device=DEVICE))

        # Calculate corresponding rate & distortion
        rate = analytical_rate(x, V, D)
        distortion = analytical_distortion(x, W, V, D)

        rate_list.append(rate)
        distortion_list.append(distortion)

    rate_expectation = sum(rate_list) / len(rate_list)
    distortion_expectation = sum(distortion_list) / len(distortion_list)

    return rate_expectation, distortion_expectation
