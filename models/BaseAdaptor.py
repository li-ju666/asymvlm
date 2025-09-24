from torch import nn


class Adaptor(nn.Module):
    def __init__(self):
        super(Adaptor, self).__init__()

    def forward(self, z_i, z_t):
        raise NotImplementedError

    def loss(self, z_i_prime, z_t_prime):
        raise NotImplementedError

    def adapt_text(self, z_t):
        raise NotImplementedError

    def adapt_image(self, z_i):
        raise NotImplementedError

    def log_likelihood(self, z_i, z_t):
        raise NotImplementedError
