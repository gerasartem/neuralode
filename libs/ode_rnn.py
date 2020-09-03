import torch
from torch import nn
from torchdiffeq import odeint

from libs.ode_net import ODENet


class ODERNN(nn.Module):
    r"""
    RNN that uses ODENet and ODE solver to patch hidden state of GRU cell on each iteration.

    Args:
        x_size (int): size of input space
        z_size (int): size of latent space (hidden state)
        hid_size (int): size of hidden layers
        method (string): method used in odeint ODE  solver (read torchdiffeq docs)
        device (torch.device): device that evaluates the model
    """
    def __init__(self, x_size, z_size, hid_size=256, method='euler', device=None):
        super(ODERNN, self).__init__()

        self.x_size = x_size
        self.z_size = z_size
        self.hid_size = hid_size
        self.method = method
        self.device = device

        if self.device is None:
            self.device = torch.device('cpu')

        self.ode = ODENet(z_size, hid_size=hid_size)
        self.gru = nn.GRUCell(x_size + 1, z_size)
        self.output = nn.Sequential(
            nn.Linear(z_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, x_size),
        )

        self._init_weights()

    def _init_weights(self):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)

        self.output.apply(init_layer)

    def forward(self, x, t, mask):
        z = torch.zeros(x.size(0), self.z_size, device=self.device)
        outputs = []

        mask = mask.unsqueeze(-1).float()
        data = torch.cat([x * mask, mask], dim=-1)

        z = self.gru(data[:, 0], z)
        outputs.append(self.output(z))
        for i in range(1, t.size(0)):
            z = odeint(self.ode, z, t[i - 1:i + 1], method=self.method, rtol=1e-3, atol=1e-4)[1]
            z = self.gru(data[:, i], z)
            outputs.append(self.output(z))

        return torch.stack(outputs, dim=1), z
