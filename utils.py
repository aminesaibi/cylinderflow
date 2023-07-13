import torch
import math
import torch.nn as nn
import os
import sys
import time
import yaml
import logging
import numpy as np
from collections import defaultdict
from multiprocessing import Process
from torch.utils.tensorboard.writer import SummaryWriter

class FCNN(nn.Module):
    """Fully connected network"""

    def __init__(self,
                 in_dim,
                 fc_sizes,
                 activation=nn.ReLU(),
                 out_activation=None,
                 bias=True,
                 weight_init_type="default"):
        super().__init__()
        self.fc_sizes = fc_sizes
        self.activ = activation
        self.out_activ = out_activation
        self.bias = bias
        self.weights_init_tpe = weight_init_type
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(in_features=in_dim,
                                     out_features=fc_sizes[0],
                                     bias=bias
                                     )
                           )

        for i in range(1, len(fc_sizes)):
            self.layers.append(nn.Linear(in_features=fc_sizes[i-1],
                                         out_features=fc_sizes[i],
                                         bias=bias
                                         )
                               )

        if weight_init_type != "default":
            weight_initializer = self.weights_init_wrapper(weight_init_type)
            self.apply(weight_initializer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activ is not None:
                x = self.activ(x)

        out = self.layers[-1](x)
        if self.out_activ is not None:
            out = self.out_activ(out)

        return out

    @staticmethod
    def weights_init_wrapper(type):
        if type == "neg_xavier_uniform":
            def weight_initializer(module):
                if isinstance(module, nn.Linear):
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    module.weight.data.uniform_(-stdv, 0)
                    if module.bias is not None:
                        module.bias.data.uniform_(-stdv, 0)

            return weight_initializer
        else:
            raise Exception(
                "Please specify a valid initialization type : \"default\" or \"neg_xavier_uniform\".")


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def runge_kutta(f, x0, u=0, y=0, dt=0.1):

    k1 = dt*f(x0, u, y)
    k2 = dt*f(x0 + 0.5*k1, u, y)
    k3 = dt*f(x0 + 0.5*k2, u, y)
    k4 = dt*f(x0 + k3, u, y)
    x1 = x0 + (k1+2*(k2+k3) + k4)/6

    return x1


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.Loader)
    return opt


def write_yaml(file, dotdict):
    d = dict(dotdict)
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)


def gen_param_rand(params: dict, param: str, lbound: float, ubound: float):

    if param not in list(params.keys()):
        return params

    if lbound > ubound:
        lbound, ubound = ubound, lbound

    val = lbound + np.random.rand()*(ubound - lbound)
    if type(params[param]) == list:
        params[param] = [val]
    else:
        params[param] = val

    return params


def cast_dtype(x: torch.Tensor, complex_val: bool, batch_dim: bool = True):
    if batch_dim:
        x = x.unsqueeze(0)
    default_dtype = torch.get_default_dtype()
    if complex_val:
        if default_dtype is torch.float32:
            dtype = torch.complex64
        elif default_dtype is torch.float64:
            dtype = torch.complex128
        else :
            raise Exception("The default dtype is neither float32 nor float64 !")
    else:
        dtype = default_dtype

    return x.to(dtype)

class Logger():
    """Logger class"""

    def __init__(self, outdir, configs, folder_tag="", hparam=None, term=True, verbose=False):

        self.verbose = verbose
        self.proc = None
        self.dic = defaultdict(list)

        tstart = str(time.time())
        tstart = tstart.replace(".", "_")
        self.directory = outdir + "/XP_" + folder_tag + tstart

        os.makedirs(self.directory, exist_ok=True)
        self.writer = SummaryWriter(self.directory)

        hparams_types = [bool, str, float, int]
        hparams = {}
        for name, config in configs.items():
            write_yaml(os.path.join(self.directory, name), config)
            if hparam is not None:
                if hparam in list(config.keys()):
                    if type(config[hparam]) == list:
                        hparams.update({(str(hparam)+str(k)): v for k,
                                       v in enumerate(config[hparam])})
                    elif type(config[hparam]) in hparams_types:
                        hparams.update({hparam: config[hparam]})
                    else:
                        print("hparam type is not supported.")
        if ((hparam is not None) and (len(hparam) != 0)):
            self.writer.add_hparams(hparams, {"metric": 0})
        else:
            if self.verbose:
                print("no hparams to log in tensorboard.")

        self.term = term

    def log(self, epoch, step_type='train'):
        if len(self.dic) == 0:
            return "no data to log"
        s = f"Epoch {epoch} : "
        for label, values in self.dic.items():
            self.writer.add_scalar(step_type+"_"+label,
                                   sum(values)*1./len(values), epoch)
            s += f"{label}:{sum(values)*1./len(values)} -- "

        self.dic.clear()
        if self.term:
            logging.info(s)

    def add(self, l):
        for label, value in l:
            self.dic[label].append(value)

    def launchTensorBoard(self, directory):
        print('tensorboard --logdir=' + directory)
        ret = os.system('tensorboard --logdir=' + directory)
        if ret != 0:
            syspath = os.path.dirname(sys.executable)
            print(os.path.dirname(sys.executable))
            ret = os.system(syspath+"/"+'tensorboard --logdir=' + directory)

    def loadTensorBoard(self):
        self.proc = Process(target=self.launchTensorBoard,
                            args=([self.directory]))
        self.proc.start()

    def stopTensorBoard(self):
        self.proc.kill()
