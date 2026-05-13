import os

import torch

import importlib

from torch.nn import init

from torch.nn.parallel import DistributedDataParallel

import sys
sys.path.append('..')

from utils.util import scandir

def dynamic_instantiation(modules, args):
    cls_type = args.net_name
    cls_ = None
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError('{} is not found.'.format(cls_type))
    return cls_(args)


def define_network(args):
    net = dynamic_instantiation(_arch_modules, args)
    return net

def init_net(net, gpu_ids=[], device=None, dist=False, init_type='normal', init_gain=0.02):
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            raise AssertionError
        net.to(device)
        if dist:
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()])
        else:
            net = torch.nn.DataParallel(net, gpu_ids)
    # init_weights(net, init_type, gain=init_gain)
    return net


def define_G(args, init_type='xavier', init_gain=0.02,):
    gpu_ids = args.gpu_ids
    device = args.device
    dist = args.dist

    net = define_network(args)
    return init_net(net, gpu_ids, device, dist, init_type, init_gain)