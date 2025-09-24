from torch import nn
from torch.nn.functional import softplus


class GaussianAdaptor(nn.Module):
    def __init__(self, *args):
        super(GaussianAdaptor, self).__init__()
        self.logvar = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, input):
        return softplus(self.logvar(input))
