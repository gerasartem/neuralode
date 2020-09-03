import torch
from torch import nn
from torchdiffeq import odeint

from libs.ode_net import ODENet
from libs.ode_rnn import ODERNN
from libs.utils import init_mask


class LatentODE(nn.Module):
    r"""
        LatentODE model implements encoder-decoder architecture. Encoder - ODERNN.
        Decoder - ODE solver with decoding net.

        Args:
            x_size (int): size of input space
            z_size (int): size of latent space (hidden state)
            hid_size (int): size of hidden layers
            noise_sigma (float): sigma used for normal distributions
            method (string): method used in odeint ODE  solver (read torchdiffeq docs)
            device (torch.device): device that evaluates the model
            reverse (bool): encode time series in reverse order
    """
    def __init__(self, x_size, z_size, hid_size=256, noise_sigma=0.01, method='euler', device=None, reverse=True):
        super(LatentODE, self).__init__()

        self.x_size = x_size
        self.z_size = z_size
        self.hid_size = hid_size
        self.noise_sigma = noise_sigma
        self.method = method
        self.device = device
        self.reverse = reverse

        if self.device is None:
            self.device = torch.device('cpu')

        self.encoder = ODERNN(x_size, z_size, hid_size=hid_size, method=method, device=self.device)

        self.mu_net = nn.Linear(z_size, z_size)
        self.sigma_net = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.Softplus(),
        )

        self.ode = ODENet(z_size, hid_size=hid_size)
        self.decoder = nn.Sequential(
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

        self.mu_net.apply(init_layer)
        self.sigma_net.apply(init_layer)
        self.decoder.apply(init_layer)

    def encode(self, x, t, mask=None):
        if mask is None:
            mask = init_mask(x)

        if self.reverse:
            _, z = self.encoder(torch.flip(x, [1]), torch.flip(t, [0]), torch.flip(mask, [1]))
        else:
            _, z = self.encoder(x, t, mask)
        mu = self.mu_net(z)
        sigma = self.sigma_net(z)

        return mu, sigma

    def decode(self, z, t):
        z = odeint(self.ode, z, t, method=self.method, rtol=1e-3, atol=1e-4).transpose(0, 1)
        return self.decoder(z)

    def prior(self, size):
        mu = torch.zeros(size, self.z_size, device=self.device)
        sigma = torch.ones(size, self.z_size, device=self.device)
        return mu, sigma

    def sample_latent(self, mu, sigma, k=1):
        """
        returns: tensor of shape (N x T x k x d)
        """
        size = sigma.size()
        size = size[:-1] + (k, size[-1])
        z = mu.unsqueeze(-2) + torch.randn(size, device=self.device) * sigma.unsqueeze(-2)
        return z

    def compute_loss(self, x_to_encode, t_to_encode, mask_to_encode=None,
                     x_to_predict=None, t_to_predict=None, mask_to_predict=None,
                     k=3, return_metrics=False):
        if mask_to_encode is None:
            mask_to_encode = init_mask(x_to_encode)
        if x_to_predict is None:
            x_to_predict = x_to_encode
            mask_to_predict = mask_to_encode
        if t_to_predict is None:
            t_to_predict = t_to_encode
        if mask_to_predict is None:
            mask_to_predict = init_mask(x_to_predict)

        prior_mu, prior_sigma = self.prior(x_to_encode.size(0))
        posterior_mu, posterior_sigma = self.encode(x_to_encode, t_to_encode, mask_to_encode)
        z = self.sample_latent(posterior_mu, posterior_sigma, k=k)

        prediction = self.decode(z, t_to_predict)

        noise_distribution = torch.distributions.Normal(prediction, torch.full_like(prediction, self.noise_sigma))
        ll = noise_distribution.log_prob(x_to_predict.unsqueeze(-2)) * mask_to_predict[..., None, None]

        latent_prior = torch.distributions.Normal(prior_mu, prior_sigma)
        latent_posterior = torch.distributions.Normal(posterior_mu, posterior_sigma)
        kl_div = torch.distributions.kl_divergence(latent_posterior, latent_prior)

        loss = kl_div.sum() - ll.sum() / k
        if return_metrics:
            mse = nn.functional.mse_loss(x_to_predict.unsqueeze(-2) * mask_to_predict[..., None, None],
                                         prediction * mask_to_predict[..., None, None])
            mae = nn.functional.l1_loss(x_to_predict.unsqueeze(-2) * mask_to_predict[..., None, None],
                                        prediction * mask_to_predict[..., None, None])
            return loss, mse, mae

        return loss

    def sample(self, size, T):
        mu, sigma = self.prior(size)
        z = self.sample_latent(mu, sigma).squeeze(-2)
        X = self.decode(z, T)
        return X

    def sample_similar(self, x, t, mask=None, custom_time=None):
        if mask is None:
            mask = init_mask(x)

        mu, _ = self.encode(x, t, mask)
        if custom_time is not None:
            sample = self.decode(mu, custom_time)
        else:
            sample = self.decode(mu, t)
        return sample
