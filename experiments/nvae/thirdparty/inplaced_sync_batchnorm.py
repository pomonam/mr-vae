# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/881c1adfcd916b6cd5de91bc343eb86aff88cc80/torch/nn/modules/batchnorm.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

from __future__ import division

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .functions import SyncBatchNorm as sync_batch_norm
from .swish import Swish as swish


class SyncBatchNormSwish(_BatchNorm):

  def __init__(self,
               num_features,
               eps=1e-5,
               momentum=0.1,
               affine=True,
               track_running_stats=True,
               process_group=None):
    super(SyncBatchNormSwish, self).__init__(num_features,
                                             eps,
                                             momentum,
                                             affine,
                                             track_running_stats)
    self.process_group = process_group
    # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
    # under supported condition (single GPU per process)
    self.ddp_gpu_size = None

  def _check_input_dim(self, input):
    if input.dim() < 2:
      raise ValueError('expected at least 2D input (got {}D input)'.format(
          input.dim()))

  def _specify_ddp_gpu_num(self, gpu_size):
    if gpu_size > 1:
      raise ValueError(
          'SyncBatchNorm is only supported for DDP with single GPU per process')
    self.ddp_gpu_size = gpu_size

  def forward(self, input):
    # currently only GPU input is supported
    # if not input.is_cuda:
    #   raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

    self._check_input_dim(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      self.num_batches_tracked = self.num_batches_tracked + 1
      if self.momentum is None:  # use cumulative moving average
        exponential_average_factor = 1.0 / self.num_batches_tracked.item()
      else:  # use exponential moving average
        exponential_average_factor = self.momentum

    # need_sync = self.training or not self.track_running_stats
    need_sync = False
    # if need_sync:
    #   process_group = torch.distributed.group.WORLD
    #   if self.process_group:
    #     process_group = self.process_group
    #   world_size = torch.distributed.get_world_size(process_group)
    #   need_sync = world_size > 1

    # fallback to framework BN when synchronization is not necessary
    if not need_sync:
      print(input.shape)
      out = F.batch_norm(input,
                         self.running_mean,
                         self.running_var,
                         self.weight,
                         self.bias,
                         self.training or not self.track_running_stats,
                         exponential_average_factor,
                         self.eps)
      return swish.apply(out)
    # else:
    #   # av: I only use it in this setting.
    #   if not self.ddp_gpu_size and False:
    #     raise AttributeError(
    #         'SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel'
    #     )
    #
    #   return sync_batch_norm.apply(input,
    #                                self.weight,
    #                                self.bias,
    #                                self.running_mean,
    #                                self.running_var,
    #                                self.eps,
    #                                exponential_average_factor,
    #                                process_group,
    #                                world_size)
