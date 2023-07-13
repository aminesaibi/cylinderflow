import math
from utils import FCNN, cast_dtype
from stochastic import OrnsteinUhlenbeck
from torch.autograd.functional import jacobian
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self,
                 x_dim: int = 3,
                 a_dim: int = 1,
                 gamma: float = 0.999,
                 fc_pi_sizes: list[int] = [128, 64, 1],
                 fc_q_sizes: list[int] = [128, 256, 32, 1],
                 pi_activation: str = 'relu',
                 value_activation: str = 'relu',
                 pi_out_activation: str = 'tanh',
                 value_out_activation: str = 'none',
                 biased_pi_linear_layers: bool = True,
                 biased_q_linear_layers: bool = True,
                 pi_weight_init_type: str = "default",
                 q_weight_init_type: str = "default",
                 action_noise_scale: float = 0.2,
                 action_noise_init_val: float = 1.0,
                 action_noise_weight_decay: float = 0.005,
                 action_noise_type: str = "OrnsteinUhlenbeck",
                 complex_dim: bool = True,
                 noise_timestep=0.1,
                 device="cpu"):
        super(Model, self).__init__()

        self.device = torch.device(device)

        if complex_dim:
            self.x_dim = x_dim*2
            self.a_dim = a_dim*2
        else:
            self.x_dim = x_dim
            self.a_dim = a_dim

        self.complex_dim = complex_dim
        self.gamma = gamma
        self.fc_pi_sizes = fc_pi_sizes
        self.fc_q_sizes = fc_q_sizes

        self.fc_pi = FCNN(self.x_dim,
                          fc_pi_sizes,
                          activation=self.activation_wrapper(pi_activation),
                          out_activation=self.activation_wrapper(pi_out_activation),
                          bias=biased_pi_linear_layers,
                          weight_init_type=pi_weight_init_type
                          ).to(self.device)

        self.fc_q = FCNN(self.x_dim+self.a_dim,
                         fc_q_sizes,
                         activation=self.activation_wrapper(value_activation),
                         out_activation=self.activation_wrapper(value_out_activation),
                         bias=biased_q_linear_layers,
                         weight_init_type=q_weight_init_type
                         ).to(self.device)

        if action_noise_type == "OrnsteinUhlenbeck":
            theta = action_noise_weight_decay
            sigma = math.sqrt(action_noise_scale*2*theta)
            self.action_noise = OrnsteinUhlenbeck(dim=self.a_dim,
                                                  complex_val=self.complex_dim,
                                                  init=action_noise_init_val,
                                                  theta=theta,
                                                  sigma=sigma,
                                                  dt=noise_timestep,
                                                  device=self.device)
        elif action_noise_type == "normal":
            self.action_noise = generate_random(dim=a_dim,
                                                dist="normal",
                                                complex_val=self.complex_dim,
                                                scale=action_noise_scale,
                                                device=self.device)
        else:
            raise Exception("Unknown action noise type.")

        self.pi_jacobian = None

    def pi(self, x, deterministic=False):
        if self.complex_dim:
            x = torch.cat([x.real, x.imag], dim=-1)
        out = self.fc_pi(x)

        if self.complex_dim:
            out = out[:, :self.a_dim//2] + 1j*out[:, self.a_dim//2:]
        if not deterministic:
            noise = self.action_noise()
            out = out + noise

        return out

    def q(self, x, a):

        if self.complex_dim:
            x = torch.cat([x.real, x.imag], dim=-1)
            a = torch.cat([a.real, a.imag], dim=-1)

        out = torch.cat([x, a], dim=-1)
        out = self.fc_q(out)

        return out

    def linearized_pi(self, x):
        """linearized pi around 0."""

        if self.pi_jacobian is not None:
            return torch.matmul(self.pi_jacobian, x)
        else:
            raise Exception("The jacobian of pi hasn't been updated yet !")

    def update(self):
        """updates model attributs that depend upon neural network parameters"""

        if self.complex_dim:
            zero_tensor = cast_dtype(torch.zeros(self.x_dim//2, device=self.device, requires_grad=False),
                                     complex_val=True,
                                     batch_dim=True
                                     )
            self.pi_jacobian = jacobian(self.pi, zero_tensor).view(self.a_dim//2,
                                                                   self.x_dim//2)
        else:
            self.pi_jacobian = jacobian(self.pi,
                                        torch.zeros(
                                            self.x_dim, device=self.device).unsqueeze(0)
                                        ).detach()\
                                         .view(self.a_dim, self.x_dim)

    @staticmethod
    def negrelu(x):
        return torch.minimum(x, torch.tensor(0))

    @staticmethod
    def negswish(x):
        return -nn.functional.silu(-x)

    def activation_wrapper(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'negrelu':
            return self.negrelu
        elif activation == 'none':
            return None
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leakyrelu':
            return nn.LeakyReLU(0.5)
        elif activation == "swish":
            return nn.SiLU()
        elif activation == "negswish":
            return self.negswish
        else:
            raise Exception("Unrocognized activation function.")
