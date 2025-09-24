from torch import nn
import torch
import math


class SphericalAdaptor(nn.Module):
    def __init__(self):
        super(SphericalAdaptor, self).__init__()
        self.etta = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input):
        return self.etta(input)

    def loss(self, etta, x):
        kappa = torch.linalg.norm(etta, dim=-1, keepdim=True)
        mu = etta / kappa
        log_ll = self.matrixwise_ll(mu, kappa, x) * self.temperature
        labels = torch.arange(etta.size(0), device=etta.device)
        l1 = nn.functional.cross_entropy(log_ll, labels)
        l2 = nn.functional.cross_entropy(log_ll.t(), labels)
        return (l1 + l2)/2

    # compute the log-likelihood between each individual image and text
    def matrixwise_ll(self, mu, kappa, x):
        raise NotImplementedError


class VMFAdaptor(SphericalAdaptor): # Von Mises-Fisher Distribution
    def matrixwise_ll(self, mu, kappa, x):
        unit_x = nn.functional.normalize(x, dim=-1)
        kappa = kappa.view(-1, 1)

        t1 = unit_x @ ((mu * kappa).t())
        t2 = self._approx_ln_c(mu.size(1), kappa.t())
        return t1 + t2

    # approximate the log-normalizer (ref: https://arxiv.org/abs/2103.15718)
    @staticmethod
    def _approx_ln_c(d, kappa):
        kappa_sqr = kappa.pow(2)
        t1 = (d-1)/4 * torch.log((d-1)/2 + torch.sqrt(((d-1)/2)**2 + kappa_sqr))
        t2 = -1/2 * torch.sqrt(((d-1)/2)**2 + kappa_sqr)
        t3 = (d-1)/4 * torch.log((d-1)/2 + torch.sqrt(((d+1)/2)**2 + kappa_sqr))
        t4 = -1/2 * torch.sqrt(((d+1)/2)**2 + kappa_sqr)
        return t1 + t2 + t3 + t4


class PSDAdaptor(SphericalAdaptor): # Power Spherical Distribution
    def matrixwise_ll(self, mu, kappa, x):
        unit_x = nn.functional.normalize(x, dim=-1)

        t1 = torch.log1p(unit_x @ mu.t()) * kappa.t()
        t2 = self._compute_ln_c(mu.size(1), kappa.t())
        return t1 + t2

    @staticmethod
    def _compute_ln_c(d, kappa):
        alpha = (d - 1) / 2 + kappa
        beta = (d - 1) / 2
        return -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
        )
