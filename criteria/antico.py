import contextlib
import copy
import os

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import batchminer
"""================================================================================================="""
ALLOWED_MINING_OPS = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM = True



class Criterion(torch.nn.Module):
    def __init__(self, opt,**kwargs):
        """
        Args:
            opt: argparse.Namespace with all training parameters.
        """
        super(Criterion, self).__init__()


        self.pars = opt
        self.n_classes = opt.n_classes

        ####
        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

        #### AntiCo_param
        self.gam1 = opt.antico_gam1
        self.gam2 = opt.antico_gam2
        self.eps = opt.antico_eps
        self.wac = opt.antico_w
        self.ac_type = opt.antico_type
        self.w_align = self.pars.loss_ac_w_align

        ####
        self.optim_dict_list = []
        self.proxies = None
        self.num_proxies = opt.n_classes
        self.embed_dim = opt.embed_dim

        self.proxies = torch.randn(self.num_proxies, self.embed_dim) / 8
        self.proxies = torch.nn.Parameter(self.proxies)

        self.optim_dict_list = [{
            'params': self.proxies,
            'lr': opt.lr * opt.loss_oproxy_lrmulti
        }]

        self.iter_count = 0
    def forward(self, batch,labels, avg_batch_features, epoch, wmup,**kwargs):
        # plot_similarity_matrix(batch,epoch,)
        """
        Args:
            batch: torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a
                    class [0,...,C-1], shape: (BS x 1)
        """
        ac_loss, ac_p = self.antico(batch, labels, self.proxies ,self.ac_type, self.n_classes)

        # # # Base Proxy Objective for global proxy alignment.
        # proxy_align_loss = self.panc(batch, self.proxies, labels)
        proxy_align_loss = self.panc(batch, self.proxies, labels)
        loss = self.w_align * proxy_align_loss + self.wac * torch.exp(ac_loss)
        return loss, ac_p, torch.nn.functional.normalize(self.proxies, dim=-1).cpu().detach().numpy()
    def print_grad(self, grad):
        print(grad)
    def pnca(self, batch, proxies, labels, dim=1):
        return self.panc(batch, proxies, labels, dim)

    def panc(self, batch, proxies, labels, dim=0):
        proxies = torch.nn.functional.normalize(proxies, dim=-1)
        batch = torch.nn.functional.normalize(batch, dim=-1)

        labels = labels.unsqueeze(1)
        u_labels, freq = labels.view(-1), None

        same_labels = (labels.T == u_labels.view(-1, 1)).to(batch.device).T
        diff_labels = (torch.arange(len(proxies)).unsqueeze(1) != labels.T).to(
            torch.float).to(batch.device).T

        w_pos_sims = -self.pars.loss_oproxy_pos_alpha * (
            batch.mm(proxies[u_labels].T) - self.pars.loss_oproxy_pos_delta)
        w_neg_sims = self.pars.loss_oproxy_neg_alpha * (
            batch.mm(proxies.T) - self.pars.loss_oproxy_neg_delta)

        pos_s = self.masked_logsumexp(w_pos_sims, mask=same_labels.type(torch.bool), dim=dim)
        neg_s = self.masked_logsumexp(w_neg_sims, mask=diff_labels.type(torch.bool), dim=dim)
        return pos_s.mean() + neg_s.mean()

    @staticmethod
    def masked_logsumexp(sims, dim=0, mask=None):
        # Adapted from https://github.com/KevinMusgrave/pytorch-metric-learning/\
        # blob/master/src/pytorch_metric_learning/utils/loss_and_miner_utils.py.
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape)
        dims[dim] = 1
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims

    def antico(self, X, Y , proxies,ac_type,num_classes=None):
        proxies = torch.nn.functional.normalize(proxies, dim=-1)
        labels = Y
        labels = labels.unsqueeze(1)
        u_labels, freq = labels.view(-1), None
        u_labels_p = torch.unique(u_labels, sorted=False)
        # print("u_labels_p:", u_labels_p)
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        if ac_type == 'batch_proxy':
            P = torch.nn.functional.normalize(proxies[u_labels_p], dim=-1).T
        elif ac_type == 'all_proxy':
            P = torch.nn.functional.normalize(proxies, dim=-1).T
        else:
            P = torch.nn.functional.normalize(proxies, dim=-1).T

        discrimn_loss_empi, discrimn_loss_empi_p, discrimn_loss_empi_p_t = self.compute_discrimn_loss(W, P)
        if ac_type == 'batch_proxy':
            total_loss_empi = self.gam2 * -discrimn_loss_empi_p_t
        elif ac_type == 'all_proxy':
            total_loss_empi = self.gam2 * -discrimn_loss_empi_p_t
        elif ac_type == 'pair':
            total_loss_empi = self.gam2 * -discrimn_loss_empi
        return total_loss_empi, discrimn_loss_empi_p

    def compute_discrimn_loss(self, W, proxies):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        # scalar = m / (p * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))

        """Empirical Discriminative Loss(proxies)."""
        p_pxy, m_pxy = proxies.shape  # Dim, P_num
        I = torch.eye(p_pxy).cuda()
        scalar = p_pxy / (m_pxy * self.eps)
        logdet_p = torch.logdet(I + self.gam1 * scalar * proxies.matmul(proxies.T))
        """Empirical Discriminative Loss(proxies)."""
        I = torch.eye(m_pxy).cuda()
        scalar = m_pxy / (p_pxy * self.eps)

        logdet_p_t = torch.logdet(I + self.gam1 * scalar * (proxies.T).matmul(proxies))

        return logdet / 2., logdet_p/2, logdet_p_t/2



