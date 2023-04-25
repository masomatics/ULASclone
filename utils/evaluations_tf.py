import torch
import torchvision
import torch.backends.cudnn as cudnn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import argparse 
import os

sys.path.append('./')
sys.path.append('./datasets')
sys.path.append('./models')

from datasets.three_dim_shapes import ThreeDimShapesDataset
from datasets.small_norb import SmallNORBDataset
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_mnist import SequentialMNIST_double
from datasets import seq_mnist as sm

import models.seqae as seqae
import models.base_networks as bn 
from models import misc_mnet as mnet

import models
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from einops import rearrange
from sklearn.metrics import r2_score
import pdb
from einops import rearrange
from utils import notebook_utils as nb
from utils import evaluations as ev
from utils import notebook_utils as nu

from torch.utils.tensorboard import SummaryWriter

import copy

import csv
import ast
from source import yaml_utils as yu

def write_tf(targdir, root='/mnt/vol21/masomatics/result/ulas'):
    device = 1
    targdir_path = os.path.join(root, targdir)

    writer = SummaryWriter(log_dir=targdir_path)

    tp=18

    Mlist = []
    config = nu.load_config(targdir_path)
    logs = nu.read_log(targdir_path)

    dataconfig = config['train_data']
    dataconfig['args']['T'] = config['T_cond'] + tp

    data = yu.load_component(dataconfig)
    train_loader = DataLoader(
            data, batch_size=config['batchsize'], shuffle=True, num_workers=config['num_workers'])

    torch.manual_seed(0)
    train_loader = DataLoader(data, batch_size=config['batchsize'], shuffle=True, num_workers=config['num_workers'])
    model_config = config['model']
    model = yu.load_component(model_config)
    iterlist = nu.iter_list(targdir_path)
    maxiter = np.max(nu.iter_list(targdir_path))
    nu.load_model(model, targdir_path, maxiter)

    #images
    writer_images(train_loader, model, config, device, tp, writer)


    #error evaluation
    allresults, targ, xnext  = ev.prediction_evaluation([targdir_path], device =0, n_cond=2, 
                                                    tp=tp, repeats=3,predictive=False,
                                                    reconstructive = False,alteration={},
                                                   mode='notebook')

    pred_error = allresults['results'][targdir_path][0]
    for i in range(len(pred_error)):
        writer.add_scalar("prediction", pred_error[i], i)
    writer.close()    


def writer_images(train_loader, model, config, device, tp, writer):
 
    images = next(iter(train_loader))
    model.eval().to(device)
    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)
    images = images.permute([0, 1, -1, 2, 3])
    images = images.to(device).float()

    reconst = False
    regconfig = config['reg']
    loss,  loss_dict = model.loss(images,  T_cond=config['T_cond'], return_reg_loss=True, reconst=reconst, regconfig=regconfig)
    T_cond = config['T_cond']
    xs = images
    return_reg_loss = False
    xs_cond = xs[:, :T_cond]
    xs_pred = model(xs_cond, return_reg_loss=return_reg_loss,
                        n_rolls=xs.shape[1] - T_cond, reconst=reconst, regconfig=regconfig)
    xs_pred = torch.sigmoid(xs_pred) 
    xs_target = xs[:, T_cond:]

    allimages = []

    num_imgs = 5
    for check_idx in range(0,num_imgs-1):
        allimages.append(xs_pred[check_idx])
        allimages.append(xs_target[check_idx])

    allimages_tensor = torch.cat(allimages)
    
    grid = torchvision.utils.make_grid(allimages_tensor, nrow=int(tp))
    writer.add_image('images', grid, 0)

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--targdir', type=str)

    args = parser.parse_args()

    write_tf(args.targdir)

