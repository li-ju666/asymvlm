from ..BaseAdaptor import Adaptor
from .modules import VMFAdaptor, PSDAdaptor
import torch


class _AsymVLM(Adaptor):
    def __init__(self, distribution):
        super(_AsymVLM, self).__init__()
        self.image_adaptor = None
        SubAdaptorDict = {
            'vmf': VMFAdaptor,
            'psd': PSDAdaptor
        }
        self.text_adaptor = SubAdaptorDict[distribution]()

    def forward(self, z_i, z_t):
        z_t_prime = self.text_adaptor(z_t)
        return z_i, z_t_prime

    def loss(self, z_i_prime, z_t_prime):
        return self.text_adaptor.loss(z_t_prime, z_i_prime)

    def adapt_text(self, z_t):
        etta = self.text_adaptor(z_t)
        # decompose etta into mu and kappa
        kappa = torch.linalg.norm(etta, dim=-1, keepdim=True)
        mu = etta / kappa
        return mu, 1/kappa.view(-1)

    def adapt_image(self, z_i):
        return z_i, None

    def log_likelihood(self, z_i, z_t):
        mu_i, _ = self.adapt_image(z_i)
        mu_t, kappa_t_inv = self.adapt_text(z_t)
        kappa_t = 1 / kappa_t_inv
        return self.text_adaptor.matrixwise_ll(mu_t, kappa_t, mu_i)


class AsymVLMPSD(_AsymVLM):
    def __init__(self):
        super(AsymVLMPSD, self).__init__('psd')


class AsymVLMVMF(_AsymVLM):
    def __init__(self):
        super(AsymVLMVMF, self).__init__('vmf')
