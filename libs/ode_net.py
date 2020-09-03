from torch import nn


class ODENet(nn.Module):
    r"""
    Simple net trained to predict gradient of time series function.

    Args:
        z_size (int): size of latent space
        hid_size (int): size of hidden layers
    """
    def __init__(self, z_size, hid_size=256):
        super(ODENet, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(z_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, z_size),
        )

        self._init_weights()

    def _init_weights(self):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)

        self.seq.apply(init_layer)

    def forward(self, t, z):
        return self.seq(z)
