import numpy
import torch
import torch.nn as nn
import einops
from torch.nn import functional as F
DEKAI_NEGATIVE = -1e8
import pdb

class Meta_Mnet(nn.Module):
    def __init__(self, batchsize=32, dim_a=16, mode='glasso',
                 normalize=False, beta=0, temperature=1.0, **kwargs):
        super().__init__()

        #self.Ms = nn.Parameter(einops.repeat(torch.eye(dim_a), 'c a -> b c a', b = batchsize).float())

        self.Ms = nn.Parameter(torch.stack([torch.eye(dim_a)] * batchsize))
        # torch.nn.init.orthogonal_(self.U)
        self.mode = mode

        self.Mnet_args = {'normalize':normalize,
                          'dim_a': dim_a,
                          'beta': beta,
                          'temperature':temperature}

    def __call__(self, H0, H1):
        #H shape : n t s a
        #H1 = H[:, 1:]
        #H0 = H[:, :-1]

        #H1hat = H0 @ self.Ms[:H0.shape[0]]
        H1hat = H0 @ self.Ms

        if self.mode == 'exact':
            loss = loss_l2_norm(H1hat, H1)
        elif self.mode == 'glasso':
            loss = loss_group_lasso_norm(H1hat, H1,
                                         **self.Mnet_args
                                         )
        else:
            raise NotImplementedError

        return loss


def loss_l2_norm(z1, z2, **kwargs):
    return torch.sqrt(torch.sum((z1 -  z2) **2))


def loss_glasso_norm(z1, z2, **kwargs):
    '''
    :param z1:  tensor b x s x a
    :param z2:  tensor b x s x a
    '''
    delta = z1- z2
    #glasso ;  sum_s |s|_2
    bs_vector = torch.sqrt(torch.sum(delta**2, axis=-1))
    b_vector = torch.sum(bs_vector, axis=-1)
    glasso_loss = torch.mean(b_vector)
    return glasso_loss

def loss_group_lasso_contrastive(z1, z2, temperature=1.0, normalize=False, beta=0,
                          **kwargs):
    '''
    :param z1:  tensor b x s x  a
    :param z2:  tensor b x s x a
    :param temperature:
    :param normalize:
    :param beta:
    :param kwargs:
    :return:
    '''

    if normalize == 1:
        z1, z2 = F.normalize(z1, p=2, dim=1), F.normalize(z2, p=2, dim=1)

    else:
        pass

    m = 2
    mask = torch.eye(z1.size(0) * m, device=z1.device)
    label0 = torch.fmod(
        z1.size(0) + torch.arange(0, m * z1.size(0), device=z1.device),
        m * z1.size(0))
    z = torch.cat([z1, z2], 0)

    '''
    Compute the group lasso of col vectors.
    If v and w are two tensors of shape S x A , then 
    this computes 
    sum_{s}    d_2(v_s - w_s ; a)  

    This will produce a logit that is computed based on the distance mat of shape B x B
    '''
    zt = z.permute([1, 0, 2])
    pre_lasso_dist = torch.cdist(zt, zt)
    logit_zz = -torch.mean(pre_lasso_dist, axis=0) / temperature
    logit_zz += mask * DEKAI_NEGATIVE

    loss = nn.CrossEntropyLoss()(logit_zz, label0)

    if beta > 0:
        denominator = log_sum_exp(logit_zz, dim=1).flatten()
        loss = loss + beta * torch.mean(denominator)

    return loss


def log_sum_exp(logit, weight=1, dim=1):

    m = torch.max(logit, dim, keepdims=True)[0]
    logit = logit - m

    sum_exp = (torch.exp(logit)).sum(dim, keepdims=True)


    return torch.log(sum_exp) + m


