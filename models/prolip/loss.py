""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license

Reference code: https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/loss.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class ProLIPLoss(nn.Module):
    def __init__(
        self,
        cache_labels=False,
        rank=0,
        world_size=1,
        bidir=True,
        use_horovod=False,
        ppcl_lambda=1,
        siglip_lambda=0,
        inclusion_alpha=0,
        inclusion_alpha_occ=0,
        inclusion_loss_eps=-10,
        inclusion_loss_scale=1000,
        vib_beta=0.00001,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

        # ProLIP loss hyperparameters
        self.siglip_lambda = siglip_lambda
        self.ppcl_lambda = ppcl_lambda
        self.vib_beta = vib_beta
        self.inclusion_alpha = inclusion_alpha
        self.inclusion_alpha_occ = inclusion_alpha_occ

        self.inclusion_loss_eps = inclusion_loss_eps
        self.inclusion_loss_scale = inclusion_loss_scale

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False, binary=False) -> torch.Tensor:
        if binary:
            labels = torch.eye(num_logits, device=device, dtype=dtype)
        else:
            labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
            if not negative_only:
                labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def get_prolip_logits(self, image_features, text_features, image_stds, text_stds, logit_scale, logit_bias):
        mu_pdist = image_features @ text_features.T
        sigma_pdist = ((torch.exp(image_stds).unsqueeze(1) + torch.exp(text_stds).unsqueeze(0))).sum(-1)
        logits = logit_scale * (mu_pdist - sigma_pdist / 2) + logit_bias

        return logits, mu_pdist, sigma_pdist

    def kl_divergence(self, mu, logsigma_sq):
        """ KL divergence between (mu, sigma) and (1, 0)
        original code: https://github.com/naver-ai/pcmepp/blob/main/pcmepp/criterions/pcmepp.py#L54-L60

        bug fix for: https://github.com/naver-ai/pcmepp/issues/4
        """
        logsigma = logsigma_sq / 2
        kl_loss = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
        if kl_loss > 10000:
            import warnings
            # XXX prevent loss exploration
            warnings.warn(f'Detected a VIB loss explosion ({kl_loss=} > 10000). Ignore the VIB loss for stability.')
            return 0
        return kl_loss

    def inclusion_test(self, mu1, logsigma_sq1, mu2, logsigma_sq2):
        """ Test if mu1, logsigma_sq1 is included in mu2, logsigma_sq2
        the test returns a large value if 1 in 2, otherwise returns a small value
        """
        eps = self.inclusion_loss_eps
        inv_sigma_sq1 = torch.exp(-logsigma_sq1 + eps)
        inv_sigma_sq2 = torch.exp(-logsigma_sq2 + eps)

        a = inv_sigma_sq1 + 0.5 * inv_sigma_sq2
        b = 2 * mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2
        c = mu1 ** 2 * inv_sigma_sq1 + 0.5 * mu2 ** 2 * inv_sigma_sq2

        return -2 * logsigma_sq1 - logsigma_sq2 - 0.5 * torch.log(a) + b ** 2 / 4 / a - c

    def inclusion_loss(self, mu1, logsigma_sq1, mu2, logsigma_sq2, is_one_included_in_two, loss_thres=100000):
        """
        is_one_included_in_two: 1 if mu1, logsigma_sq1 is included in mu2, logsigma_sq2 otherwise -1
        """
        logit = self.inclusion_test(mu1, logsigma_sq1, mu2, logsigma_sq2) - self.inclusion_test(mu2, logsigma_sq2, mu1, logsigma_sq1)
        loss = -F.logsigmoid(self.inclusion_loss_scale * logit * is_one_included_in_two)
        inclusion_loss = loss.mean()

        if inclusion_loss.abs() > loss_thres:
            import warnings
            warnings.warn(f'Detected a inlcusion loss explosion ({inclusion_loss=} > {loss_thres}). Ignore the inclusion loss for stability.')
            return 0
        return inclusion_loss

    def _loss(self, image_features, text_features, logit_scale=None, logit_bias=None,
              shift=None, negative_scale=None, negative_only=False, text_mu=None, text_sigma=None,
              masked_image_features=None, masked_text_features=None):
        loss = 0
        loss_dict = {}

        image_mu, image_sigma = image_features["mean"], image_features["std"]
        if text_features is not None:
            # NOTE For world_size > 1 (See L290-292 for more details)
            text_mu, text_sigma = text_features["mean"], text_features["std"]

        labels = self.get_ground_truth(
            image_mu.device,
            image_mu.dtype,
            image_mu.shape[0],
            negative_only=negative_only,
        )

        # Deterministic loss (for ablation study only)
        # NOTE We use SigLIP loss for deterministic loss because ProLIP is based on pairwise contrastive loss by PCME++
        if self.siglip_lambda > 0:
            logits = self.get_logits(image_mu, text_mu, negative_scale, shift)
            siglip_loss = self.siglip_lambda * -F.logsigmoid(labels * logits).sum() / image_mu.shape[0]
            loss += siglip_loss
            loss_dict["siglip"] = siglip_loss

        # Probabilistic Pairwise Contrastive Loss (PPCL)
        logits, mu_pdist, sigma_pdist = self.get_prolip_logits(image_mu, text_mu, image_sigma, text_sigma, logit_scale, logit_bias)
        pcmepp_loss = self.ppcl_lambda * -F.logsigmoid(labels * logits).sum() / image_mu.shape[0]
        loss += pcmepp_loss
        loss_dict["ppcl"] = pcmepp_loss

        # NOTE uncomment this if you need to log them
        # log_dict = {}
        # log_dict["LOG_mu_pdist"] = mu_pdist.mean()
        # log_dict["LOG_sigma_pdist"] = sigma_pdist.mean()
        # log_dict["LOG_img_sigma"] = torch.exp(image_sigma).mean()
        # log_dict["LOG_txt_sigma"] = torch.exp(text_sigma).mean()

        # Inclusion loss (original feature \subset masked feature)
        if masked_image_features is not None:
            n_masked_samples = masked_image_features["mean"].size()[0]

            img_inclusion_loss = self.inclusion_loss(
                image_mu[:n_masked_samples], image_sigma[:n_masked_samples],
                masked_image_features["mean"], masked_image_features["std"], 1
            )
            img_inclusion_loss *= self.inclusion_alpha_occ
            loss += img_inclusion_loss
            loss_dict["img_inclusion"] = img_inclusion_loss

            txt_inclusion_loss = self.inclusion_loss(
                text_mu[:n_masked_samples], text_sigma[:n_masked_samples],
                masked_text_features["mean"], masked_text_features["std"], 1
            )
            txt_inclusion_loss *= self.inclusion_alpha_occ
            loss += txt_inclusion_loss
            loss_dict["txt_inclusion"] = txt_inclusion_loss

        # Inclusion loss (image feature \subset text feature)
        if self.inclusion_alpha > 0:
            inclusion_loss = self.inclusion_loss(
                image_mu, image_sigma, text_mu, text_sigma, 1
            )
            inclusion_loss *= self.inclusion_alpha
            loss += inclusion_loss
            loss_dict["inclusion"] = inclusion_loss

        # VIB loss
        if self.vib_beta > 0:
            vib_loss = self.vib_beta * (
                self.kl_divergence(image_mu, image_sigma) + self.kl_divergence(text_mu, text_sigma))
            loss += vib_loss
            loss_dict["vib"] = vib_loss
        return loss

    def forward(self, image_features, text_features,
                logit_scale=1.0, logit_bias=0.0, shift=None, negative_scale=None,
                masked_image_features=None, masked_text_features=None,
                output_dict=False):
        loss = self._loss(
            image_features, text_features, logit_scale, logit_bias, shift, negative_scale,
            masked_image_features=masked_image_features,
            masked_text_features=masked_text_features)
        # print(f"ProLIP loss: {loss.keys()}")
        return loss