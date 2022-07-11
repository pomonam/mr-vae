import os
import random
import torch.nn.functional as F
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector


def get_conv(conv_f,
             in_chan,
             out_chan,
             kernel=4,
             stride=2,
             padding=0,
             norm=False,
             relu=False):
    """
        Helper function which builds a conv_f -> norm -> relu module and skips
        the bias in conv_f when needed
        - Note that conv_f can be conv2d or convtranspose2d
    """
    layer = [conv_f(in_chan, out_chan, kernel, stride, padding, bias=not norm)]
    if norm:
        layer.append(nn.BatchNorm2d(out_chan))
    if relu:
        layer.append(nn.ReLU())
    return layer


def get_conv_layers(conv_f, channels, ker_sizes, strides, paddings, norms, relus):
    """
        Takes in channels, ker_sizes, strides and paddings as lists
        where len(ker_sizes) == len(strides) == len(paddings) == len(channels) - 1
        returns of nn modules to splat into nn.Sequential
    """
    layer = []
    for i in range(len(channels) - 1):
        in_chan, out_chan = channels[i], channels[i + 1]
        kernel_size, stride, padding, norm, relu = ker_sizes[i], strides[i], paddings[i], \
                                                    norms[i], relus[i]
        layer += get_conv(conv_f,
                          in_chan,
                          out_chan,
                          kernel=kernel_size,
                          stride=stride,
                          padding=padding,
                          norm=norm,
                          relu=relu)
    return layer


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def sample_beta(shape, sample_range, device, apply_exp=False):
    a = sample_range[0]
    b = sample_range[1]

    if apply_exp:
        a = math.log(a)
        b = math.log(b)

    if a == b:
        betas = torch.ones(shape[0], 1).to(device) * a
    else:
        betas = torch.FloatTensor(shape[0], 1).uniform_(a, b).to(device)
    return betas


def load_activation(name):
    activations = {
        "id": lambda z: z,
        "relu": F.relu,
        "tanh": torch.tanh,
    }
    return activations[name]


def make_parameter(shape, device):
    new_param = nn.Parameter(torch.Tensor(*shape))
    new_param.data = new_param.to(device)
    return new_param


def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    names = []
    for name, _ in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append(name)
    return orig_params, names


def make_functional_with_params(model):
    orig_params = tuple(model.parameters())
    # Remove all the parameters in the model
    i = 0
    names = []
    for name, p in list(model.named_parameters()):
        del_attr(model, name.split("."))
        names.append((name, orig_params[i]))
        del p
        i += 1
    return names


def load_weights(model, names, params, as_params=False):
    for name, p in zip(names, params):
        if not as_params:
            set_attr(model, name.split("."), p)
        else:
            set_attr(model, name.split("."), torch.nn.Parameter(p))


def vector_to_param_tuple(vec, params):
    if not isinstance(vec, torch.Tensor):
        raise TypeError("expected torch.Tensor, but got: {}".format(
            torch.typename(vec)))

    pointer = 0
    split_tensors = []
    for param in params:
        num_param = param.numel()
        split_tensors.append(vec[pointer:pointer + num_param].view_as(param))
        pointer += num_param
    return tuple(split_tensors)


def tensor_to_tuple(vec, parameters):
    if not isinstance(vec, torch.Tensor):
        raise TypeError("expected torch.Tensor, but got: {}".format(
            torch.typename(vec)))
    pointer = 0
    split_tensors = []
    for param in parameters:
        num_param = param.numel()
        split_tensors.append(vec[pointer:pointer + num_param].view_as(param))
        pointer += num_param
    return tuple(split_tensors)


def param_tuple_to_vector(params):
    return parameters_to_vector(params)


def param_scale(params, scale):
    return tuple(scale * p for p in params)


def param_add(params_a, params_b):
    return tuple(a + b for a, b in zip(params_a, params_b))


def param_sub(params_a, params_b):
    return tuple(a - b for a, b in zip(params_a, params_b))


def param_dot(params_a, params_b):
    return sum((a * b).sum() for a, b in zip(params_a, params_b))


def param_e_norm(params):
    return torch.sqrt(param_dot(params, params))


def param_sqe_norm(params):
    return param_dot(params, params)


def param_e_dist(params_a, params_b):
    return torch.sqrt(param_sqe_norm(param_sub(params_a, params_b)))


def gnhvp(g, f, primals, tangents):
    z, r_z = torch.autograd.functional.jvp(f, primals, tangents)
    r_gz = torch.autograd.functional.hvp(g, (z, ), (r_z, ))[1]
    vjp = torch.autograd.functional.vjp(f, primals, r_gz)[1]
    return vjp


def _select_seed_randomly(min_seed_value=0, max_seed_value=255):
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        print(f"No correct seed found, seed set to {seed}")

    if not min_seed_value <= seed <= max_seed_value:
        print(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
