""" Uncertainty modules
Reference code:
    PIENet in
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn


class GaussianAdaptor(nn.Module):
    def __init__(self, emb_dim, *args):
        super(GaussianAdaptor, self).__init__()
        self.logvar = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
        )

    def forward(self, x):
        # make sure the input is non-negative
        x = x/torch.linalg.norm(x, dim=-1, keepdim=True)
        return x, self.logvar(x)
