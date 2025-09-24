from ..BaseAdaptor import Adaptor
from .modules import GaussianAdaptor
import torch


class PFE(Adaptor):
    def __init__(self, *args):
        super(PFE, self).__init__()
        self.image_adaptor = GaussianAdaptor()
        self.text_adaptor = GaussianAdaptor()

    def forward(self, z_i, z_t):
        var_i = self.image_adaptor(z_i)
        var_t = self.text_adaptor(z_t)
        return (z_i, var_i), (z_t, var_t)

    def loss(self, z_i_prime, z_t_prime):
        mu_i, var_i = z_i_prime
        mu_t, var_t = z_t_prime

        t1 = (mu_i - mu_t).pow(2)
        t2 = 1.0/(var_i + var_t)
        t3 = (var_i + var_t).log()

        loss = ((t1 * t2 + t3).sum(dim=-1)).mean()
        return loss

    def adapt_text(self, z_t):
        var_t = self.text_adaptor(z_t)
        return z_t, var_t.mean(dim=-1)

    def adapt_image(self, z_i):
        var_i = self.image_adaptor(z_i)
        return z_i, var_i.mean(dim=-1)

    def log_likelihood(self, z_i, z_t):
        mu_i, _ = self.adapt_image(z_i)
        mu_t, _ = self.adapt_text(z_t)

        # normalize the vectors
        mu_i = mu_i / torch.linalg.norm(mu_i, dim=-1, keepdim=True)
        mu_t = mu_t / torch.linalg.norm(mu_t, dim=-1, keepdim=True)

        return mu_i @ (mu_t.t())
