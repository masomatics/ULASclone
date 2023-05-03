import numpy as np
import torch
import torch.nn as nn
from models import dynamics_models
import torch.nn.utils.parametrize as P
from models.dynamics_models import LinearTensorDynamicsLSTSQ, MultiLinearTensorDynamicsLSTSQ, HigherOrderLinearTensorDynamicsLSTSQ, LinearTensorDynamicsLSTSQ_inner
from models.base_networks import ResNetEncoder, ResNetDecoder, Conv1d1x1Encoder, MLP_iResNet, LinearNet, MLP
from models import base_networks as bn
from einops import rearrange, repeat
from utils.clr import simclr
import pdb
from sklearn.metrics import r2_score
import copy
from utils import misc
import einops
import torch.autograd as autograd
from models import misc_mnet as mnet
from collections import OrderedDict


class SeqAELSTSQ(nn.Module):
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            detachM=0,
            *args,
            **kwargs):
        super().__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = LinearTensorDynamicsLSTSQ(alignment=alignment)
        self.detachM = detachM
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def _encode_base(self, xs, enc):
        #batch, time, ch, h, w
        shape = xs.shape
        x = torch.reshape(xs, (shape[0] * shape[1], *shape[2:]))
        H = enc(x)
        #batch x time x flatten_dimen
        H = torch.reshape(
            H, (shape[0], shape[1], *H.shape[1:]))
        return H

    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        # batch x time x flatten_dimen
        H = torch.reshape(
            H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])

        return H

    def phi(self, xs):
        return self._encode_base(xs, self.enc.phi)

    #WARNING: get_M is not directly used in the decoding process.
    #M is always computed in self.dynamics_model
    def get_M(self, xs):
        dyn_fn = self.dynamics_fn(xs)
        Mstar = dyn_fn.M
        if self.detachM == 1:
            Mstar = Mstar.detach()
        return Mstar

    def decode(self, H):
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(torch.linalg.inv(self.change_of_basis),
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        n, t = H.shape[:2]
        if hasattr(self, "pidec"):
            H = rearrange(H, 'n t d_s d_a -> (n t) d_a d_s')
            H = self.pidec(H)
        else:
            H = rearrange(H, 'n t d_s d_a -> (n t) (d_s d_a)')
        x_next_preds = self.dec(H)
        x_next_preds = torch.reshape(
            x_next_preds, (n, t, *x_next_preds.shape[1:]))
        return x_next_preds

    def dynamics_fn(self, xs, return_loss=False, fix_indices=None):
        H = self.encode(xs)
        return self.dynamics_model(H, return_loss=return_loss, fix_indices=fix_indices)

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False, regconfig={}):
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, reconst=reconst, regconfig=regconfig)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
            #losses = (loss_bd(dyn_fn.M, self.alignment),
            #          loss_orth(dyn_fn.M), loss_internal_T)
        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]

        loss = torch.mean(
            torch.sum((xs_target - torch.sigmoid(xs_pred)) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss

    def __call__(self, xs_cond, return_reg_loss=False, n_rolls=1, fix_indices=None, reconst=False,
                 regconfig = {}):
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a

        H = self.encode(xs_cond)

        # ==Esitmate dynamics==  M IS APPLIED HERE
        ret = self.dynamics_model(
            H, return_loss=return_reg_loss, fix_indices=fix_indices, do_detach=self.detachM)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, losses = ret
        else:
            fn = ret

        if self.predictive:
            H_last = H[:, -1:]
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)

        else:
            H_last = H[:, :1]
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        #repeated application of prediction
        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)

        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        x_preds = self.decode(H_preds)

        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds

class SeqAELSTSQ_inner(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            alignment=False,
            kernel_size=3,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            t_c=2,
            dmode='default',
            mnet_mlp=False,
            detachM=0,
            normalize=3,
            change_of_basis=False,
            inner_args={},
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.alignment = alignment
        self.dmode = dmode
        self.initial_scale_M = 0.01
        self.normalize = normalize
        self.inner_args = inner_args
        self.inner_args['dim_a'] = dim_a
        self.detachM = detachM
        self.enc = ResNetEncoder(
            dim_a * dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.t_c = t_c
        #self.M_net = mnet.Meta_Mnet(**self.inner_args)


        '''
        Use the nonparametric M_net/ exact that computes the argmin 
        over the loop given a set of H := (n, t, s, a) 
        This nonparameteric M_net outputs the set of matrices (n, a, a) 
        '''

        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, n_blocks=n_blocks,
            bottom_width=bottom_width)

        self.dynamics_model = LinearTensorDynamicsLSTSQ_inner(alignment=alignment,
                                                              inner_args=self.inner_args)

        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)





#iResNet class for So3 use
class SeqAELSTSQ_so3Net(SeqAELSTSQ):

    def __init__(
            self,
            dim_a,
            dim_m,
            **kwargs):

        super().__init__(dim_a, dim_m, **kwargs)

        self.enc = MLP(in_dim=dim_a * dim_m, out_dim=dim_a * dim_m)
        self.dec = MLP(in_dim=dim_a * dim_m, out_dim=dim_a * dim_m)

    #No sigmoid. ALso, match the dimension of the pred
    def loss(self, xs,
             return_reg_loss=True,
             T_cond=2,
             reconst=False,
             R2=False):
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond, reconst=reconst)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
            losses = {'reg_bd': reg_losses[0],
                      'reg_orth': reg_losses[1],
                      'loss_internal': reg_losses[2]}
            #losses = (loss_bd(dyn_fn.M, self.alignment),
            #          loss_orth(dyn_fn.M), loss_internal_T)
        if reconst:
            xs_target = xs
        else:
            xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        if R2 == True:
            loss = r2_score(xs_target.reshape(len(xs_target), -1),
                            xs_pred.reshape(len(xs_pred), -1).detach())
        else:
            loss = torch.mean(
                torch.sum((xs_target - xs_pred) ** 2, axis=[1, 2]))
        return (loss, reg_losses) if return_reg_loss else loss



class SeqAELSTSQ_iResNet(SeqAELSTSQ_so3Net):
    def __init__(
            self,
            dim_a,
            dim_m,
            **kwargs):
        super().__init__(dim_a, dim_m, **kwargs)

        self.enc = MLP_iResNet(in_dim=dim_a * dim_m)
        self.dec = MLP_iResNet(in_dim=dim_a * dim_m)



#Linear class for so3 use (This is implicitly calling MLP in SeqAELSTSQ_so3Net,
#it must be changed)
class SeqAELSTSQ_LinearNet(SeqAELSTSQ_so3Net):

    def __init__(
            self,
            dim_a,
            dim_m,
            **kwargs):
        super().__init__(dim_a,dim_m, **kwargs)

        self.enc = LinearNet(in_dim=dim_a * dim_m, out_dim=dim_a * dim_m)
        self.dec = LinearNet(in_dim=dim_a * dim_m, out_dim=dim_a * dim_m)


class SeqAEHOLSTSQ(SeqAELSTSQ):
    # Higher order version of SeqAELSTSQ
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            n_order=2,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = HigherOrderLinearTensorDynamicsLSTSQ(
            alignment=alignment, n_order=n_order)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def __call__(self, xs, n_rolls=1, fix_indices=None, return_reg_loss=False):
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a
        H = self.encode(xs)

        # ==Esitmate dynamics==
        ret = self.dynamics_model(
            H, return_loss=return_reg_loss, fix_indices=fix_indices)
        if return_reg_loss:
            # fn is a map by M_star. Loss is the training external loss
            fn, Ms, losses = ret
        else:
            fn, Ms = ret

        if self.predictive:
            Hs_last = [H[:, -1:]] + [M[:, -1:] for M in Ms]
            array = np.arange(n_rolls)
        else:
            Hs_last = [H[:, :1]] + [M[:, :1] for M in Ms]
            array = np.arange(xs.shape[1] + n_rolls - 1)

        # Create prediction for the unseen future
        H_preds = []
        for _ in array:
            Hs_last = fn(Hs_last)
            H_preds.append(Hs_last[0])
        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        x_preds = self.decode(H_preds)
        if return_reg_loss:
            return x_preds, losses
        else:
            return x_preds

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        if reconst:
            raise NotImplementedError
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, return_reg_loss=return_reg_loss,
                       n_rolls=xs.shape[1] - T_cond)
        if return_reg_loss:
            xs_pred, reg_losses = xs_pred
        xs_target = xs[:, T_cond:] if self.predictive else xs[:, 1:]
        loss = torch.mean(
                torch.sum((xs_target - torch.sigmoid(xs_pred)) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss


class SeqAEMultiLSTSQ(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            alignment=False,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            change_of_basis=False,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            K=8,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.K = K
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)
        self.dynamics_model = MultiLinearTensorDynamicsLSTSQ(
            dim_a, alignment=alignment, K=K)
        if change_of_basis:
            self.change_of_basis = nn.Parameter(
                torch.empty(dim_a, dim_a))
            nn.init.eye_(self.change_of_basis)

    def get_blocks_of_M(self, xs):
        M = self.get_M(xs)
        blocks = []
        for k in range(self.K):
            dim_block = self.dim_a // self.K
            blocks.append(M[:, k*dim_block:(k+1)*dim_block]
                          [:, :, k*dim_block:(k+1)*dim_block])
        blocks_of_M = torch.stack(blocks, 1)
        return blocks_of_M


class SeqAENeuralM(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            alignment=False,
            kernel_size=3,
            predictive=True,
            bottom_width=4,
            n_blocks=3,
            t_c = 2,
            dmode='default',
            mnet_mlp=False,
            normalize=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.predictive = predictive
        self.alignment = alignment
        self.dmode = dmode
        self.initial_scale_M = 0.01
        self.normalize = normalize

        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.t_c = t_c

        if dmode in ['orth', 'mix'] and mnet_mlp in [1, 2]:
            if mnet_mlp == 1:
                self.M_net = bn.ResNetMLP(in_dim=t_c * dim_m*dim_a, out_dim=dim_a*dim_a,  n_blocks=n_blocks,
                                          hidden_multiple=0.2)
            if mnet_mlp == 2:
                self.M_net = bn.MLP(in_dim=t_c * dim_m*dim_a, out_dim=dim_a*dim_a,  n_blocks=n_blocks, initmode = 'default',
                                    hidden_multiple=0.2)
        else:
            self.M_net = ResNetEncoder(
                dim_a*dim_a, k=k, kernel_size=kernel_size, n_blocks=n_blocks)

        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, n_blocks=n_blocks, bottom_width=bottom_width)

    def dynamics_fn(self, xs, verbose=False):
        M, H = self.get_M(xs)
        dyn_fn = dynamics_models.LinearTensorDynamicsLSTSQ.DynFn(M)
        if verbose:
            return dyn_fn, M, H
        else:
            return dyn_fn

    # dmode changes what is fed to get M
    def get_M(self, xs):
        dev = xs.device
        H = self.encode(xs)
        if self.dmode == 'lstq':
            raise NotImplementedError
        else:
            if self.dmode == 'delta':
                xsinput = xs[:, 1:] - xs[:, :-1]
                xsinput = rearrange(xsinput, 'n t c h w -> n (t c) h w')
            elif self.dmode == 'innerp':
                #shape n t s a
                H0 = H[:, 1:]
                H1 = H[:, :-1]
                #this is still quite huge; very ad hoc at this point
                xsinput = (H0 @ H1.permute([0, 1, 3, 2])).detach().reshape([-1, 16 ,256, 16])
            elif self.dmode == 'innerpout':
                H0 = H[:, 1:]
                H1 = H[:, :-1]
                xsinput = (H0.permute([0, 1, 3, 2]) @ H1).detach()
            #Perturbation with orthogonal/mixing transformation
            elif self.dmode in ['orth', 'mix'] :
                # n t s a
                device = xs.device
                n, t, s, a = H.shape
                if self.dmode == 'orth':
                    P = torch.tensor(
                        np.random.normal(size=[n, 1, s, s])).float().to(device)
                    U, _, V = torch.linalg.svd(P)
                    P = U @ V
                else:
                    P= torch.tensor(np.random.normal(size=[n, 1, s, s])).float().to(dev)
                Pmat = P.tile(dims = (1, t, 1, 1)).float()
                #(n t s s) @ (n t s a). s part is being transformed
                xsinput = Pmat @ H.detach()
            elif self.dmode == 'plainh':
                xsinput = H.detach()
            elif self.dmode == 'latent':
                xsinput = H
            #baseline Neural M Star
            elif self.dmode == 'default':
                xsinput = rearrange(xs, 'n t c h w -> n (t c) h w')
            else:
                print(f"""{self.dmode} has no implementation""" )
                raise NotImplementedError
            M = self.M_net(xsinput)
        M = rearrange(M, 'n (a_1 a_2) -> n a_1 a_2', a_1=self.dim_a)
        M = self.initial_scale_M * M
        return M, H

    def __call__(self, xs,
                 n_rolls=1,
                 return_reg_loss=False,
                 return_Ms=False,
                 reconst=False, regconfig={}):
        # ==Esitmate dynamics==
        fn, M, H = self.dynamics_fn(xs, verbose=True)
        dev = xs.device

        #either predict from the last element or from the first element
        if reconst:
            #H = self.encode(xs)
            if self.predictive:
                H_last = H[:, -1:]
            else:
                H_last = H[:, :1]
        else:
            H_last = self.encode(xs[:, -1:] if self.predictive else xs[:, :1])

        #placeholder foro H_preds. If predictive is false, the H_preds is to include 1:t_p time sequence
        if self.predictive:
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs.shape[1] + n_rolls - 1)

        # Create prediction for the unseen future. When predictive is false, this future includes the Tc part.
        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
        x_preds = self.decode(H_preds)

        if return_reg_loss:
            losses = {}
            losses['reg_bd'] = dynamics_models.loss_bd(fn.M, self.alignment) if regconfig['reg_bd'] != 'None' else 0

            losses['reg_orth'] = dynamics_models.loss_orth(fn.M) if regconfig['reg_orth'] != 'None' else 0

            losses['reg_comm'] = self.M_commutability(fn.M) if regconfig['reg_comm'] != 'None' else 0

            losses['reg_inv'] = self.M_algebraic_inv(M, H) if regconfig['reg_inv'] != 'None' else 0
            return x_preds, losses
        else:
            return x_preds

    def M_commutability(self, M, sample_size=10):
        loss = 0
        for _ in range(sample_size):
            id1, id2 = np.random.choice(len(M), 2, replace=False)
            comm = (M[id1] @ M[id2]) - (M[id2] @ M[id1])
            loss = loss + torch.sum(comm**2)
        return loss

    #sample_size: number of random sampling of P

    #To be used for regularization
    def M_algebraic_inv(self, M, H,  sample_size=2):

        # n t s a
        Hs = H.detach()
        M0 = M
        dev = M.device
        loss = 0
        for _ in range(sample_size):
            # n t a s
            Hs_t = Hs.permute([0, 1, 3, 2])
            ds = Hs_t.shape[-1]
            if self.dmode == 'mix':
                Pmat = torch.tensor(np.random.normal(size=[ds, ds])).float().to(dev)
            elif self.dmode == 'orth':
                Pmat = misc.get_orthmat(ds, ds).to(dev)
            else:
                return torch.tensor([0]).float().to(dev)
            Hs_tP = Hs_t @ Pmat
            Hs_P = Hs_tP.permute([0,1,3,2]).detach()
            M1 =self.M_net(Hs_P)
            M1 = rearrange(M1, 'n (a_1 a_2) -> n a_1 a_2', a_1=self.dim_a)
            loss = loss + torch.sum((M1 - M0)**2)
        return loss



class SeqAENeuralM_latentPredict(SeqAENeuralM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def conduct_prediction(self, xs, n_rolls=1, T_cond=2, reconst=True,
                           verbose= False):
        predout = self.compute_H_preds(xs, n_rolls=n_rolls, T_cond=T_cond, reconst=reconst, verbose=verbose)
        if verbose == True:
            H_preds, _, _, M, _ = predout
            x_preds = torch.sigmoid(self.decode(H_preds))
            return x_preds, H_preds
        else:
            H_preds, _ = predout
            x_preds = torch.sigmoid(self.decode(H_preds))
            return x_preds

    def compute_H_preds(self, xs, n_rolls=1, T_cond=2, reconst=True,
                        verbose=False):
        xs_cond = xs[:, :T_cond]
        # H is H_cond.
        fn, M, H = self.dynamics_fn(xs_cond, verbose=True)

        # either predict from the last element or from the first element
        if reconst:
            # H = self.encode(xs)
            if self.predictive:
                H_target = self.encode(xs)
                H_last = H[:, -1:]
            else:
                H_target = self.encode(
                    xs[:, T_cond:]) if self.predictive else self.encode(xs)
                H_last = H[:, :1]
        else:
            H_target = self.encode(
                xs[:, T_cond:]) if self.predictive else self.encode(xs)
            H_last = self.encode(xs[:, -1:] if self.predictive else xs[:, :1])

        # placeholder foro H_preds. If predictive, the H_preds is to include 1:t_p time sequence
        if self.predictive:
            H_preds = [H] if reconst else []
            array = np.arange(n_rolls)
        else:
            H_preds = [H[:, :1]] if reconst else []
            array = np.arange(xs_cond.shape[1] + n_rolls - 1)

        # Create prediction for the unseen future. When predictive, this future includes the Tc part.
        for _ in array:
            H_last = fn(H_last)
            H_preds.append(H_last)
        H_preds = torch.cat(H_preds, axis=1)
        if verbose == True:
            return H_preds, H_target, fn, M, H
        else:
            return H_preds, H_target

    def __call__(self, xs,
                 n_rolls=1,
                 T_cond = 3,
                 return_reg_loss=False,
                 return_Ms=False,
                 reconst=False, regconfig={}):
        # reconstructed x
        x_reconst = self.decode(self.encode(xs))
        dev = xs.device

        H_preds, H_target, fn, M, H = self.compute_H_preds(xs, n_rolls=n_rolls, T_cond=T_cond, reconst=reconst, verbose=True)

        if return_reg_loss:
            losses = {}
            losses['reg_bd'] = dynamics_models.loss_bd(fn.M, self.alignment) if regconfig['reg_bd'] != 'None' else torch.tensor([0]).to(dev)

            losses['reg_orth'] = dynamics_models.loss_orth(fn.M) if regconfig['reg_orth'] != 'None' else torch.tensor([0]).to(dev)

            losses['reg_comm'] = self.M_commutability(fn.M) if regconfig['reg_comm'] != 'None' else torch.tensor([0]).to(dev)

            losses['reg_inv'] = self.M_algebraic_inv(M, H) if regconfig['reg_inv'] != 'None' else torch.tensor([0]).to(dev)

            losses['reg_latent'] = self.latent_error(H_preds, H_target) if regconfig['reg_latent'] != 'None' else torch.tensor([0]).to(dev)

            losses['reg_obs'] = self.obs_error(H_preds, xs) if regconfig['reg_obs'] != 'None' else torch.tensor([0]).to(dev)

            return x_reconst, losses
        else:
            return x_reconst

    def latent_error(self, H_preds, H_target):
        latent_e = torch.mean((H_preds - H_target)**2)
        return latent_e

    def obs_error(self, H_preds, xs, reconst=True, T_cond=2):

        if reconst:
            # H = self.encode(xs)
            if self.predictive:
                x_target = xs
            else:
                x_target = xs[:, T_cond:] if self.predictive else xs
        else:
            x_target = xs[:, T_cond:] if self.predictive else xs

        latent_e = torch.sum((torch.sigmoid(self.decode(H_preds)) - x_target)**2)
        return latent_e

    def normalize_isotypic_copy(self, H):
        isotype_norm = torch.sqrt(torch.sum(H**2, axis=self.normalize, keepdims=True))
        H = H/isotype_norm
        return H

    #encoding  function with isotypic column normalization
    def encode(self, xs):
        H = self._encode_base(xs, self.enc)
        # batch x time x flatten_dimen
        H = torch.reshape(
            H, (H.shape[0], H.shape[1], self.dim_m, self.dim_a))
        if hasattr(self, "change_of_basis"):
            H = H @ repeat(self.change_of_basis,
                           'a1 a2 -> n t a1 a2', n=H.shape[0], t=H.shape[1])
        H = self.normalize_isotypic_copy(H)

        return H

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False, regconfig={}):
        xs_reconst = self(xs, return_reg_loss=return_reg_loss, T_cond= T_cond,
                       n_rolls=xs.shape[1] - T_cond, reconst=reconst, regconfig=regconfig)
        if return_reg_loss:
            xs_reconst, reg_losses = xs_reconst
        xs_target = xs
        loss = torch.mean(
            torch.sum((xs_target - torch.sigmoid(xs_reconst)) ** 2, axis=[2, 3, 4]))
        return (loss, reg_losses) if return_reg_loss else loss




class SeqAENeuralM_comm(SeqAENeuralM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def dynamics_fn(self, xs, verbose=False):
        M = self.get_M(xs)
        bsize = len(M)
        #swapM = copy.copy(M[torch.arange(-bsize // 2, M.shape[0] - bsize // 2)])
        swapM = M[torch.arange(-bsize // 2, M.shape[0] - bsize // 2)]
        #replace M with  M'^-1  M  M'  where M' are Mperm
        M = torch.linalg.solve(swapM, M@swapM)

        dyn_fn = dynamics_models.LinearTensorDynamicsLSTSQ.DynFn(M)
        if verbose:
            return dyn_fn, M
        else:
            return dyn_fn




class SeqAENeuralTransition(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            ch_x=3,
            k=1.0,
            kernel_size=3,
            T_cond=2,
            bottom_width=4,
            n_blocks=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.T_cond = T_cond
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.ar = Conv1d1x1Encoder(dim_out=dim_a)
        self.dec = ResNetDecoder(
            ch_x, k=k, kernel_size=kernel_size, bottom_width=bottom_width, n_blocks=n_blocks)

    def loss(self, xs, return_reg_loss=False, T_cond=2, reconst=False):
        assert T_cond == self.T_cond
        xs_cond = xs[:, :T_cond]
        xs_pred = self(xs_cond, n_rolls=xs.shape[1] - T_cond, reconst=reconst)
        xs_target = xs if reconst else xs[:, T_cond:]
        loss = torch.mean(
            torch.sum((xs_target - torch.sigmoid(xs_pred)) ** 2, axis=[2, 3, 4]))
        if return_reg_loss:
            return loss, [torch.Tensor(np.array(0, dtype=np.float32)).to(xs.device)] * 3
        else:
            return loss

    def get_M(self, xs):
        T = xs.shape[1]
        xs = rearrange(xs, 'n t c h w -> (n t) c h w')
        H = self.enc(xs)
        H = rearrange(H, '(n t) c -> n (t c)', t=T)
        return H

    def __call__(self, xs, n_rolls=1, reconst=False):
        # ==Esitmate dynamics==
        H = self.encode(xs)

        array = np.arange(n_rolls)
        H_preds = [H] if reconst else []
        # Create prediction for the unseen future
        for _ in array:
            H_pred = self.ar(rearrange(H, 'n t s a -> n (t a) s'))
            H_pred = rearrange(
                H_pred, 'n (t a) s-> n t s a', t=1, a=self.dim_a)
            H_preds.append(H_pred)
            H = torch.cat([H[:, 1:], H_pred], dim=1)
        H_preds = torch.cat(H_preds, axis=1)
        # Prediction in the observation space
        return self.decode(H_preds)


class CPC(SeqAELSTSQ):
    def __init__(
            self,
            dim_a,
            dim_m,
            k=1.0,
            kernel_size=3,
            temp=0.01,
            normalize=True,
            loss_type='cossim',
            n_blocks=3,
            *args,
            **kwargs):
        super(SeqAELSTSQ, self).__init__()
        self.dim_a = dim_a
        self.dim_m = dim_m
        self.normalize = normalize
        self.temp = temp
        self.loss_type = loss_type
        self.enc = ResNetEncoder(
            dim_a*dim_m, k=k, kernel_size=kernel_size, n_blocks=n_blocks)
        self.ar = Conv1d1x1Encoder(dim_out=dim_a*dim_m)

    def __call__(self, xs):
        H = self.encode(xs)  # [n, t, s, a]

        # Create prediction for the unseen future
        H = rearrange(H, 'n t s a -> n (t a) s')

        # Obtain c in CPC
        H_pred = self.ar(H)  # [n a s]
        H_pred = rearrange(H_pred, 'n a s -> n s a')

        return H_pred

    def get_M(self, xs):
        T = xs.shape[1]
        xs = rearrange(xs, 'n t c h w -> (n t) c h w')
        H = self.enc(xs)
        H = rearrange(H, '(n t) c -> n (t c)', t=T)
        return H

    def loss(self, xs, return_reg_loss=True, T_cond=2, reconst=False):
        T_pred = xs.shape[1] - T_cond
        assert T_pred == 1
        # Encoded Latent. Num_ts x len_ts x  dim_m x dim_a
        H = self.encode(xs)  # [n, t, s, a]

        # Create prediction for the unseen future
        H_cond = H[:, :T_cond]
        H_cond = rearrange(H_cond, 'n t s a -> n (t a) s')

        # Obtain c in CPC
        H_pred = self.ar(H_cond)  # [n a s]
        H_pred = rearrange(H_pred, 'n a s -> n s a')

        H_true = H[:, -1]  # n s a
        H_true = rearrange(H_true, 'n s a -> n (s a)')
        H_pred = rearrange(H_pred, 'n s a -> n (s a)')
        loss = simclr([H_pred, H_true], self.temp,
                      normalize=self.normalize, loss_type=self.loss_type)
        if return_reg_loss:
            reg_losses = [torch.Tensor(np.array(0, dtype=np.float32))]*3
            return loss, reg_losses
        else:
            return loss
