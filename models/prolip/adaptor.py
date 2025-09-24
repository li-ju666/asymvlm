""" PCME model base code

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch.nn as nn
import torch
from .modules import GaussianAdaptor
from .loss import ProLIPLoss


class ProLIP(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, emb_dim=512):
        super(ProLIP, self).__init__()

        self.img_enc = GaussianAdaptor(emb_dim=emb_dim)
        self.txt_enc = GaussianAdaptor(emb_dim=emb_dim)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ProLIPLoss()

    def forward(self, z_i, z_t):
        image_output = self.img_enc(z_i)
        caption_output = self.txt_enc(z_t)

        return (
            image_output[0],
            image_output[1],
            caption_output[0],
            caption_output[1],)

    def loss(self, z_i, z_i_logsigma, z_t, z_t_logsigma):
        i_emb = {
            'mean': z_i,
            'std': z_i_logsigma,
        }
        t_emb = {
            'mean': z_t,
            'std': z_t_logsigma,
        }
        return self.loss_func(i_emb, t_emb)

    def adapt_text(self, z_t):
        text_output = self.txt_enc(z_t)
        return text_output[0], text_output[1].exp().mean(dim=1)

    def adapt_image(self, z_i):
        image_output = self.img_enc(z_i)
        return image_output[0], image_output[1].exp().mean(dim=1)

    def log_likelihood(self, z_i, z_t):
        mu_i, _ = self.adapt_image(z_i)
        mu_t, _ = self.adapt_text(z_t)
        # normalize the vectors
        mu_i = mu_i / torch.linalg.norm(mu_i, dim=-1, keepdim=True)
        mu_t = mu_t / torch.linalg.norm(mu_t, dim=-1, keepdim=True)
        return mu_i @ (mu_t.t())
