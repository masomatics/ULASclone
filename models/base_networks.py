import numpy as np
import torch
from torch import nn
from models.resblock import Block, Conv1d1x1Block, Invertible_Resblock_Fc
from einops.layers.torch import Rearrange
from einops import repeat
from utils import misc
from torch.nn import functional as F
import copy
import pdb
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py


class Conv1d1x1Encoder(nn.Sequential):
    def __init__(self,
                 dim_out=16,
                 dim_hidden=128,
                 act=nn.ReLU()):
        super().__init__(
            nn.LazyConv1d(dim_hidden, 1, 1, 0),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Conv1d1x1Block(dim_hidden, dim_hidden, act=act),
            Rearrange('n c s -> n s c'),
            nn.LayerNorm((dim_hidden)),
            Rearrange('n s c-> n c s'),
            act,
            nn.LazyConv1d(dim_out, 1, 1, 0)
        )


class ResNetEncoder(nn.Module):
    def __init__(self,
                 dim_latent=1024,
                 k=1,
                 act=nn.ReLU(),
                 kernel_size=3,
                 n_blocks=3):
        super().__init__()
        self.phi = nn.Sequential(
            nn.LazyConv2d(int(32 * k), 3, 1, 1),
            *[Block(int(32 * k) * (2 ** i), int(32 * k) * (2 ** (i+1)), int(32 * k) * (2 ** (i+1)),
                    resample='down', activation=act, kernel_size=kernel_size) for i in range(n_blocks)],
            nn.GroupNorm(min(32, int(32 * k) * (2 ** n_blocks)),
                         int(32 * k) * (2 ** n_blocks)),
            act)
        self.linear = nn.LazyLinear(
            dim_latent) if dim_latent > 0 else lambda x: x

    def __call__(self, x):
        h = x
        h = self.phi(h)
        h = h.reshape(h.shape[0], -1)
        h = self.linear(h)
        return h


class ResNetDecoder(nn.Module):
    def __init__(self, ch_x, k=1, act=nn.ReLU(), kernel_size=3, bottom_width=4, n_blocks=3):
        super().__init__()
        self.bottom_width = bottom_width
        self.linear = nn.LazyLinear(int(32 * k) * (2 ** n_blocks))
        self.net = nn.Sequential(
            *[Block(int(32 * k) * (2 ** (i+1)), int(32 * k) * (2 ** i), int(32 * k) * (2 ** i),
                    resample='up', activation=act, kernel_size=kernel_size, posemb=True) for i in range(n_blocks-1, -1, -1)],
            nn.GroupNorm(min(32, int(32 * k)), int(32 * k)),
            act,
            nn.Conv2d(int(32 * k), ch_x, 3, 1, 1)
        )

    def __call__(self, x):
        x = self.linear(x)
        x = repeat(x, 'n c -> n c h w',
                   h=self.bottom_width, w=self.bottom_width)
        x = self.net(x)
        return x


'''
Invertible ResNet, stacked
'''
class MLP_iResNet(nn.Module):
    def __init__(self, in_dim, nonlin='elu', n_power_iter=2, n_resblocks=3):
        super(MLP_iResNet, self).__init__()
        self.nonlin = nonlin
        self.n_resblocks = n_resblocks
        self.in_dim = in_dim
        self.blocklist= []
        for k in range(self.n_resblocks):
            self.blocklist.append(Invertible_Resblock_Fc(in_dim,
                                                    nonlin=nonlin,
                                                    n_power_iter=n_power_iter,
                                                    lip_const=0.97))
        if n_resblocks > 0:
            self.network = nn.Sequential(*self.blocklist)
        else:
            self.network = lambda x: x

    def __call__(self, x):
        x = x.flatten(start_dim=1)
        return self.network(x)


class LinearNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, **kwargs):
        super(LinearNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.network = nn.Linear(in_features=in_dim, out_features=out_dim)

    def __call__(self, x):
        return self.network(x)


class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=3,
                 num_layer=3,
                 activation=nn.ELU,
                 hidden_multiple = 6,
                 initmode='cond',
                 dospec=False,
                 **kwargs):

        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = []
        for k in range(num_layer):
            if k == 0:
                dimin = in_dim
                dimout = in_dim * hidden_multiple
            else:
                dimin = in_dim * hidden_multiple
                dimout = in_dim * hidden_multiple
            linlayer = nn.Linear(in_features=dimin,
                                         out_features=dimout)
            if initmode == 'cond':
                initialize_linear(linlayer, thresh=2.0)
            elif initmode == 'default':
                nn.init.orthogonal_(linlayer.weight.data)
                nn.init.uniform_(linlayer.bias.data)
            else:
                raise NotImplementedError
            if dospec == True:
                snorm = misc.specnorm(nn.Parameter(linlayer.weight.data))
                linlayer.weight.data  = nn.Parameter(0.97 * linlayer.weight.data/snorm)

            self.layers.append(linlayer)
            self.layers.append(activation())
        linlayer = nn.Linear(in_features=dimout,
                                         out_features=self.out_dim)


        self.layers.append(linlayer)

        self.network = nn.Sequential(*self.layers)

    def __call__(self, x):
        return self.network(x)


def initialize_linear(mylayer, thresh = 2.0):

    nn.init.xavier_uniform_(mylayer.weight.data)
    #nn.init.normal_(mylayer.weight.data)

    weightmat  = mylayer.weight.data

    while torch.linalg.cond(weightmat) > thresh*weightmat.shape[0]:
        print('initializing loop')
        nn.init.xavier_uniform_(mylayer.weight.data)
        #nn.init.normal_(mylayer.weight.data)

        weightmat = mylayer.weight.data

    print('condition passed')
    mylayer.weight.data = nn.Parameter(weightmat)




class Radial_sine(nn.Module):
    def __init__(self,  scale=3., freq=6. , ztrf=False, **kwargs):
        super().__init__()
        self.scale = scale
        self.freq = freq
        self.ztrf = ztrf

    def __call__(self, dat):
        dat = copy.deepcopy(dat)
        original_shape = dat.shape
        dat = dat.reshape([-1, 6])
        #num_blocks = int(tensordim/3)

        for pos in range(2):

            x = dat[:, 3*pos + 0]
            y = dat[:, 3*pos + 1]
            z = dat[:, 3*pos + 2]
            rad_dis = self.freq * torch.arctan(x / y)

            if self.ztrf == True:
                rad_disz =  self.freq * torch.arcsin(z)
            else:
                rad_disz =  np.pi/2


            x = x * (np.sin(rad_dis) * np.sin(rad_disz) + self.scale)
            y = y * (np.sin(rad_dis) * np.sin(rad_disz) + self.scale)
            z = z * (np.sin(rad_dis) * np.sin(rad_disz) + self.scale)

            dat[:, 3*pos + 0] = x
            dat[:, 3*pos + 1] = y
            dat[:, 3*pos + 2] = z

        dat = dat.reshape(-1, original_shape[-1])
        return dat
