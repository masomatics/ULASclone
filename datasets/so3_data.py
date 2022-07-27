import numpy as np
import torch
from torch import nn
from models.base_networks import MLP_iResNet, LinearNet, MLP
from torch.utils.data import DataLoader
import pickle
import pdb
import os
from einops import rearrange


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



class SO3rotationSequence():
    # Rotate around z axis only.

    def __init__(
            self,
            train=True,
            T=3,
            data_filename='so3dat_sphere_iResNet.pt',
            provide_label=False,
            **kwargs):

        self.T = T
        self.T_max = 20
        self.mode = 'train' if train == True else 'test'
        #number of distinct so3 rotations
        data_path = os.path.join(dat_root, 'so3', data_filename)

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
                   latent_dimen=3,
                   tensor_dim=10,
                   latent_mode='sphere',
                   max_rot_speed=0.8,
                   max_scale_speed=0,
                   datsize=10000,
                   num_blocks=2,
                   embedding_path='',
                   save_datapath='so3',
                   video_length=15,
                   match_dimen=False,
                   shared_transition=False,
                   ):
    latent_data = torch.tensor(
        np.random.normal(size=(datsize, num_blocks*latent_dimen, tensor_dim)))
    if latent_mode == 'ResNet':
        pass
    elif latent_mode == 'sphere':
        for pos in range(num_blocks):
            norms = torch.sqrt(torch.sum(latent_data[:,3*pos:3*(pos+1),:]**2, axis=1, keepdims=True))
            latent_data[:,3*pos:3*(pos+1),:] = latent_data[:,3*pos:3*(pos+1),:] / norms
    else:
        raise NotImplementedError

    if embed_fxn_mode == 'Identity':
        embed_fxn = lambda x: x
    elif embed_fxn_mode == 'iResNet':
        embed_fxn = MLP_iResNet(in_dim=3*num_blocks*tensor_dim)

    elif embed_fxn_mode == 'Linear':
        indim= 3 * num_blocks*tensor_dim
        outdim = 3 * num_blocks*tensor_dim
        embed_fxn = LinearNet(in_dim=indim, out_dim=outdim)

    elif embed_fxn_mode == 'MLP':
        indim= 3 * num_blocks*tensor_dim
        outdim = 3* num_blocks*tensor_dim
        embed_fxn = MLP(in_dim=indim,
                        out_dim=outdim)
    else:
        raise NotImplementedError

    if len(embedding_path) > 0:
        embed_dict = torch.load(embedding_path)['state_dict']
        embed_fxn.load_state_dict(embed_dict, strict=False)
    else:
        pass

    train_loader = DataLoader(
        latent_data, batch_size=1, shuffle=False,
        num_workers=2)

    dataset = []
    trans_params = []
    for idx, latent_vec in enumerate(train_loader):
        #if shared_transition == True:


        latent_video, trans_param = create_latent_video(latent_vec[0],
                                           num_blocks,
                                           max_rot_speed,
                                           max_scale_speed,
                                           video_length)
        so3_video = embed_fxn(latent_video).detach()
        # if match_dimen == True:
        #     so3_video = rearrange(so3_video, 'b w -> b 1 1 w')
        dataset.append(so3_video)
        trans_params.append(trans_param)
        if idx % 1000 == 0:
            print(f"""{idx} videos processed.""")

    dataset = torch.stack(dataset)

    dataset_with_label = {'data': dataset, 'trans':trans_params}

    data_save_path = os.path.join(dat_root, save_datapath , f"""so3dat_{latent_mode}_{embed_fxn_mode}.pt""")
    #trans_save_path = os.path.join(dat_root, save_datapath , f"""so3dat_{latent_mode}_{embed_fxn_mode}_trans.pt""")
    model_save_path = os.path.join(dat_root, save_datapath , f"""so3dat_{latent_mode}_{embed_fxn_mode}_model.pt""")


    torch.save(dataset_with_label, data_save_path)
    #if type(embed_fxn) == nn.Module:
    torch.save(embed_fxn.state_dict(), model_save_path)
    #torch.save(trans_params, trans_save_path)


    print(f"""dataset saved at {data_save_path}""")
    print(f"""MODEL saved at {model_save_path}""")



def create_latent_video(latent_vec,
                        num_blocks,
                        max_rot_speed,
                        max_scale_speed,
                        T):

    trans_param = {}
    blocks = []
    for k in range(num_blocks):
        rot_angle = torch.tensor(np.random.uniform(0, 2 * max_rot_speed, size=1))
        rot_axis = torch.tensor(np.random.uniform(0, 1, size=3))
        rot_axis = rot_axis /torch.sqrt(torch.sum(rot_axis**2))
        blocks.append({'angle': rot_angle, 'axis': rot_axis})

    scale_speed = np.random.uniform(-max_scale_speed, max_scale_speed)

    trans_param['blocks'] = blocks
    trans_param['scale_speed'] = scale_speed
    latent_video = create_video_from_trans_param(latent_vec, T, trans_param)

    return latent_video, trans_param

def create_video_from_trans_param(latent_vec, T, trans_param):
    blocks = trans_param['blocks']
    scale_speed = trans_param['scale_speed']
    num_blocks = len(blocks)

    latent_outputs = []
    scale = scale_speed + 1
    for t in range(T):
        v_t = np.zeros(latent_vec.shape)
        for k in range(num_blocks):
            rodrigues_mat = rodrigues_rotation(blocks[k]['axis'], t * blocks[k]['angle'])
            v_t[(k*3):(k+1)*3, :] = scale * np.dot(rodrigues_mat, latent_vec[(k*3):(k+1)*3, :]).astype(np.float32)
        v_t = torch.tensor(v_t)
        latent_outputs.append(v_t)
    latent_video = torch.stack(latent_outputs).float()
    return latent_video










