import numpy as np
import torch
from torch import nn
from models.base_networks import MLP_iResNet, LinearNet, MLP
from models import base_networks as bn
from torch.utils.data import DataLoader
import pickle
import pdb
import os
from einops import rearrange
import argparse
import pickle
import copy

#dataset root
dat_root = '/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/'

#rotatoin in the angle of theta about the axis k
#https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

def rodrigues_rotation(w, theta):
    norm_w = torch.sqrt(torch.sum(w**2))
    w = w /norm_w
    bw = torch.tensor([[0, -w[2], w[1]],[ w[2], 0, -w[0]], [ -w[1], w[0], 0]])
    R = torch.eye(3) + torch.sin(theta)*bw + (1- torch.cos(theta)) * torch.matmul(bw, bw)
    return R

def two_d_rotation(theta):
    R  = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]] )
    return R



class SO3rotationSequence():
    # Rotate around z axis only.

    def __init__(
            self,
            train=True,
            T=3,
            data_filename='so3dat_sphere_iResNet.pt',
            provide_label=False,
            datamode= 'so3',
            **kwargs):

        self.T = T
        self.T_max = 20
        self.mode = 'train' if train == True else 'test'
        #number of distinct so3 rotations
        data_path = os.path.join(dat_root, datamode, data_filename)

        data_with_labels = torch.load(data_path)
        self.data = data_with_labels['data']
        self.labels = data_with_labels['trans']
        self.provide_label = provide_label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.provide_label == True:
            return self.data[i, :self.T, :], self.labels[i]
        else:
            return self.data[i, :self.T, :]


def create_dataset(embed_fxn_mode='Identity',
                   tensor_dim=10,
                   latent_mode='sphere',
                   max_rot_speed=0.8,
                   max_scale_speed=0,
                   datsize=10000,
                   num_blocks=2,
                   embedding_path='',
                   save_datapath='so3',
                   video_length=15,
                   shared_transition=False,
                   recreate_model=False,
                   seed=0,
                   changeb=False,
                   PmatNet=''
                   ):
    torch.manual_seed(seed)
    np.random.seed(seed)
    do_save_model = False

    datamode=save_datapath
    datpath = os.path.join(dat_root, datamode)
    if os.path.exists(datpath) == False:  os.mkdir(datpath)

    if datamode == 'so3':
        latent_dimen = 3
    elif datamode == 'so2':
        latent_dimen = 2
    else:
        raise NotImplementedError

    #creation of latent_data
    latent_data = torch.tensor(
        np.random.normal(size=(datsize, tensor_dim, num_blocks*latent_dimen)))
    if latent_mode == 'ResNet':
        pass
    elif latent_mode == 'sphere':
        for pos in range(num_blocks):
            norms = torch.sqrt(torch.sum(latent_data[:,:,latent_dimen*pos:latent_dimen*(pos+1)]**2, axis=2, keepdims=True))
            latent_data[:,:,latent_dimen*pos:latent_dimen*(pos+1)] = latent_data[:,:,latent_dimen*pos:latent_dimen*(pos+1)] / norms
    else:
        raise NotImplementedError


    if embed_fxn_mode == 'Identity':
        embed_fxn = lambda x: x
    elif embed_fxn_mode == 'iResNet':
        indim = latent_dimen*num_blocks*tensor_dim
        embed_fxn = MLP_iResNet(in_dim=indim)

    elif embed_fxn_mode == 'Linear':
        indim= latent_dimen * num_blocks*tensor_dim
        outdim = latent_dimen * num_blocks*tensor_dim
        embed_fxn = LinearNet(in_dim=indim, out_dim=outdim)

    elif embed_fxn_mode == 'MLP':
        indim= latent_dimen * num_blocks*tensor_dim
        outdim = latent_dimen * num_blocks*tensor_dim
        embed_fxn = MLP(in_dim=indim,
                        out_dim=outdim)
    elif embed_fxn_mode == 'sine':
        embed_fxn = bn.Radial_sine()
    elif embed_fxn_mode == 'sinetwo':

        embed_fxn = bn.Radial_sine(ztrf=1, scale=3.)
    else:
        raise NotImplementedError


    if len(embedding_path) > 0 and recreate_model == False:

        if embed_fxn_mode in ['sine', 'sinetwo']:
            file = open(embedding_path, 'rb')
            embed_fxn = pickle.load(file)

        else:
            embed_dict = torch.load(embedding_path)
            embed_fxn.load_state_dict(embed_dict, strict=False)
            print(f"""{embedding_path} successfully loaded.""")
    elif recreate_model == True:
        answer = input("create a new embedding model? y/n \n")
        if answer == 'y':
            pass
        else:
            quit()
        do_save_model = True
    else:
        print(f"""{embedding_path} does not exist. Set --recreate=1 and run again.""")
        raise NotImplementedError

    train_loader = DataLoader(
        latent_data, batch_size=1, shuffle=False,
        num_workers=2)

    dataset = []
    latent_dataset = []
    trans_params = []
    trans_param_shared = random_trans_param(num_blocks, max_rot_speed, max_scale_speed, datamode)

    #filename addendums
    addendum = '_shared_trans' if shared_transition == True else ''
    if changeb == True: addendum = addendum + '_latentP'

    #PmatNet for changing P along orbit
    if len(PmatNet) == 0:
        pass
    else:
        netpath = os.path.join(datpath, PmatNet)
        PmatNet = pickle.load(netpath)


    for idx, latent_vec in enumerate(train_loader):
        if shared_transition == True:
            trans_param = trans_param_shared
        else:
            trans_param = random_trans_param(num_blocks, max_rot_speed,
                                             max_scale_speed, datamode=datamode)
        latent_video, trans_mats= create_video_from_trans_param(latent_vec[0], video_length, trans_param,
                                                                 datamode=datamode, PmatNet=PmatNet)

        latent_dataset.append(latent_video)



        latent_video_input = rearrange(copy.deepcopy(latent_video), 't d_s d_a -> t (d_s d_a)')
        so3_video = embed_fxn(latent_video_input).detach()

        dataset.append(so3_video)
        trans_params.append(trans_mats)

        if idx % 1000 == 0:
            print(f"""{idx} videos processed.""")
    dataset = torch.stack(dataset)
    latent_dataset = torch.stack(latent_dataset)

    dataset_with_label = {'data': dataset, 'trans':trans_params, 'latent':latent_dataset}

    data_save_path = os.path.join(datpath, f"""{datamode}dat_{latent_mode}_{embed_fxn_mode}{addendum}.pt""")
    model_save_path = os.path.join(datpath, f"""{datamode}dat_{latent_mode}_{embed_fxn_mode}_model.pt""")


    torch.save(dataset_with_label, data_save_path)
    #if type(embed_fxn) == nn.Module:
    if do_save_model == True:

        if embed_fxn_mode in ['sine', 'sinetwo']:
            model_save_path = model_save_path.replace('.pt', '.pkl')
            filehandler = open(model_save_path, "wb")
            pickle.dump(embed_fxn, filehandler)

        else:
            torch.save(embed_fxn.state_dict(), model_save_path)
        print(f"""MODEL saved at {model_save_path}""")

    #torch.save(trans_params, trans_save_path)


    print(f"""dataset saved at {data_save_path}""")


def random_trans_param(num_blocks, max_rot_speed, max_scale_speed,
                       datamode = 'so3'):
    trans_param = {}
    if datamode == 'so3':
        blocks = []
        for k in range(num_blocks):
            rot_angle = torch.tensor(
                np.random.uniform(0, np.pi/2. * max_rot_speed, size=1))
            rot_axis = torch.tensor(np.random.uniform(0, 1, size=3))
            rot_axis = rot_axis / torch.sqrt(torch.sum(rot_axis ** 2))
            blocks.append({'angle': rot_angle, 'axis': rot_axis})

        scale_speed = np.random.uniform(-max_scale_speed, max_scale_speed)

        trans_param['blocks'] = blocks
        trans_param['scale_speed'] = scale_speed
    elif datamode == 'so2':
        blocks = []
        for k in range(num_blocks):
            rot_angle = torch.tensor(
                np.random.uniform(0, np.pi/2. * max_rot_speed, size=1))
            blocks.append(rot_angle)
        trans_param['blocks'] = blocks
    else:
        raise NotImplementedError

    return trans_param


def create_video_from_trans_param(latent_vec, T, trans_param, datamode='so3', PmatNet=''):
    blocks = trans_param['blocks']
    num_blocks = len(blocks)

    latent_outputs = []
    trans_mats  = []
    dim = latent_vec.shape[1]
    for t in range(T):
        block_diag_mat = torch.zeros([dim, dim]).float()
        # #if Changeb == True, then change the basis inside each sequence.
        # if type(PmatNet) == str:
        #     Pdep = torch.eye(block_diag_mat.shape[0])
        # else:
        #     Pdep = PmatNet(latent_vec)

        if datamode == 'so3':
            for k in range(num_blocks):
                rodrigues_mat = torch.tensor(rodrigues_rotation(blocks[k]['axis'], t * blocks[k]['angle']))
                block_diag_mat[(k*3):(k+1)*3, (k*3):(k+1)*3] = rodrigues_mat.float()
        elif datamode == 'so2':
            for k in range(num_blocks):
                rotation_matrix = torch.tensor(two_d_rotation(t * blocks[k]))
                block_diag_mat[(k*2):(k+1)*2, (k*2):(k+1)*2] = rotation_matrix.float()
        else:
            raise NotImplementedError
        #block_diag_mat = torch.linalg.solve(Pdep, block_diag_mat @ Pdep)
        v_t = (latent_vec.float() @ block_diag_mat).float()
        latent_outputs.append(v_t)
        trans_mats.append(block_diag_mat)
    latent_video = torch.stack(latent_outputs).float()
    mats_video = torch.stack(trans_mats).float()
    return latent_video, mats_video





if __name__ == '__main__':
    # Loading the configuration arguments from specified config path
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embed')
    parser.add_argument('-s', '--size', type=int, default=10000)
    parser.add_argument('-t', '--shared', type=int, default=0)
    parser.add_argument('-w', '--warning', action='store_true')
    parser.add_argument('-d', '--datamode', default='so3')
    parser.add_argument('-nb', '--num_blocks', type=int, default=2)
    parser.add_argument('-r', '--recreate', type=int, default=0)
    parser.add_argument('-cb', '--changeb', type=int, default=0)




    args = parser.parse_args()
    #datamode = 'so3'


    embedding_filename = f"""{args.datamode}dat_sphere_{args.embed}_model.pt"""
    embedding_path = os.path.join(dat_root, args.datamode, embedding_filename)

    if args.embed in ['sine']:
        embedding_path = embedding_path.replace('.pt', '.pkl')

    if not os.path.exists(embedding_path):
        print(f"""{embedding_path} does not exist. This will be created""")
        embedding_path = ''
    create_dataset(embed_fxn_mode=args.embed,
                   tensor_dim=10,
                   latent_mode='sphere',
                   max_rot_speed=0.8,
                   max_scale_speed=0,
                   datsize=args.size,
                   num_blocks=args.num_blocks,
                   embedding_path=embedding_path,
                   save_datapath=args.datamode,
                   video_length=15,
                   shared_transition=args.shared,
                   seed=0,
                   recreate_model=args.recreate,
                   changeb=args.changeb
                   )




