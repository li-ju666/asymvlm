from ..BaseAdaptor import Adaptor
from .modules import BayesCap_MLP
from .loss import TempCombLoss
import torch


class ProbVLM(Adaptor):
    def __init__(self, *args):
        super(ProbVLM, self).__init__()
        self.img_BayesCap = BayesCap_MLP(
            inp_dim=512, out_dim=512, hid_dim=1024,
            num_layers=3, p_drop=0.0)
        self.txt_BayesCap = BayesCap_MLP(
            inp_dim=512, out_dim=512, hid_dim=1024,
            num_layers=3, p_drop=0.0)
        self.Cri = TempCombLoss()

    def forward(self, i_features, t_features):
        
        # print('dbg', i_features.shape, t_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        ret_i = (img_mu, img_1alpha, img_beta, i_features)
        ret_t = (txt_mu, txt_1alpha, txt_beta, t_features)
        return ret_i, ret_t

    def loss(self, z_i_prime, z_t_prime):
        T1, T2 = 1.0, 5e-2
        cross_modal_lambda = 1e-4
        # unpack the outputs
        img_mu, img_1alpha, img_beta, xfI = z_i_prime
        txt_mu, txt_1alpha, txt_beta, xfT = z_t_prime

        loss_i = self.Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
        loss_t = self.Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
        #cross modal terms
        loss_i4t = self.Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
        loss_t4i = self.Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
        loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)

        return loss

    def adapt_text(self, z_t):
        mu, alpha, beta = self.txt_BayesCap(z_t)
        return mu, 1/self._get_GGuncer(alpha, beta).mean(dim=-1)

    def adapt_image(self, z_i):
        mu, alpha, beta = self.img_BayesCap(z_i)
        return mu, 1/self._get_GGuncer(alpha, beta).mean(dim=-1)

    def log_likelihood(self, z_i, z_t):
        mu_i, _ = self.adapt_image(z_i)
        mu_t, _ = self.adapt_text(z_t)

        mu_i = mu_i / torch.linalg.norm(mu_i, dim=-1, keepdim=True)
        mu_t = mu_t / torch.linalg.norm(mu_t, dim=-1, keepdim=True)

        return mu_i @ (mu_t.t())

    @staticmethod
    def _get_GGuncer(x_alpha, x_beta):
        a = 1/(x_alpha + 1e-5)
        a = torch.clip(a, min=1e-4, max=5)
        b = x_beta + 0.1
        b = torch.clip(b, min=0.1, max=5)
        u = (a**2)*torch.exp(torch.lgamma(3/b))/torch.exp(torch.lgamma(1.0/b))
        return u
