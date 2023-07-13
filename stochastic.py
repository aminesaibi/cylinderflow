import torch
from utils import cast_dtype
import math
from typing import Union


class OrnsteinUhlenbeck:
    """
    dX_t = theta.(mu - X_t).dt + sigma.dB_t
    """

    def __init__(self, dim, complex_val=False, mu=0.0, theta=0.005, sigma=0.1, init=0.0, dt=0.1, device="cpu"):
        self.device = device
        self.dim = dim
        self.complex = complex_val
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = torch.ones(self.dim, device=device)*init

    def __call__(self):
        """X_{t+1} = X_{t} + theta.(mu - X_t).dt + sigma.DeltaB"""
        drift = self.theta * (self.mu - self.X)
        # brownian increment
        delta_B = torch.randn(
            len(self.X), device=self.device)*math.sqrt(self.dt)
        self.X = self.X + drift*self.dt + self.sigma*delta_B
        if self.complex:
            return (self.X[:self.dim//2] + 1j*self.X[self.dim//2:])/math.sqrt(2)
        else:
            return self.X


class generate_random:
    """generates a random array.

    Args :
     dim : dimension of the array.
     distrib : \"uniform\" or \"normal\". "
     scale : bound for the components for uniform dist and variance std for
     normal.
     complex_val : bool for complex values.

    Returns:
      numpy array of random values.
    """

    def __init__(self,
                 dim: Union[int, tuple[int, int]] = 1,
                 scale: float = 1.0,
                 dist: str = "uniform",
                 complex_val: bool = False,
                 device: str = "cpu",
                 dt: float = 1.0):
        if type(dim) is tuple:
            self.size = dim
            self.dim = dim[1]
        elif type(dim) is int:
            self.size = (dim,)
            self.dim = dim
        else:
            raise Exception(
                f"argument dim of type {type(dim)} is unsopported !")
        self.scale = scale
        self.dist = dist
        self.complex_val = complex_val
        self.device = device
        self.dt = dt

    def __call__(self):
        if self.scale == 0:
            return cast_dtype(torch.zeros(self.size, device=self.device),
                              self.complex_val,
                              batch_dim=False
                              )
        if self.dist == "uniform":
            if not self.complex_val:
                x = (self.scale*math.sqrt(12.0/float(self.dim))) * \
                    (torch.rand(self.size, device=self.device) - 0.5)
            else:
                arg = 2*math.pi * \
                    torch.rand(self.size, device=self.device) - math.pi
                signed_modulus = (self.scale*math.sqrt(12.0/float(self.dim))) * \
                    (torch.rand(self.size, device=self.device) - 0.5)
                x = signed_modulus*torch.exp(1j*arg)
        elif self.dist == "normal":
            if not self.complex_val:
                x = (self.scale/math.sqrt(self.dim)) * \
                    torch.normal(0, 1.0, self.size).to(self.device)
            else:
                x = (self.scale/math.sqrt(2*self.dim))*(torch.normal(0, 1.0, self.size, device=self.device) +
                                                        1j*torch.normal(0, 1.0, self.size, device=self.device))
        else:
            raise Exception(
                "currently supported distributions are \"uniform\" and \"normal\".")
        return x
