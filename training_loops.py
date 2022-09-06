import math
import torch
from torch import nn
import pytorch_pfn_extras as ppe
from utils.clr import simclr
from utils.misc import freq_to_wave
from tqdm import tqdm
import pdb

def loop_seqmodel(manager, model, optimizer, train_loader, config, device):
    while not manager.stop_trigger:
        for images in train_loader:
            with manager.run_iteration():
                reconst = True if manager.iteration < config['training_loop']['args']['reconst_iter'] else False
                if manager.iteration >= config['training_loop']['args']['lr_decay_iter']:
                    optimizer.param_groups[0]['lr'] = config['lr']/3.
                else:
                    optimizer.param_groups[0]['lr'] = config['lr']
                model.train()
                if type(images) == list:
                    images = torch.stack(images)
                    images = images.transpose(1, 0)
                images = images.to(device)
                regconfig = config['reg']
                #loss,  (loss_bd, loss_orth, loss_comm) = model.loss(images,  T_cond=config['T_cond'], return_reg_loss=True, reconst=reconst)
                loss,  loss_dict = model.loss(images,  T_cond=config['T_cond'], return_reg_loss=True, reconst=reconst, regconfig=regconfig)

                optimizer.zero_grad()

                comm_const = regconfig['reg_comm'] if regconfig['reg_comm'] != 'None' else 0
                inv_const = regconfig['reg_inv'] if regconfig['reg_inv'] != 'None' else 0
                reclat_const = regconfig['reg_latent'] if regconfig['reg_latent'] != 'None' else 0


                loss = loss + comm_const * loss_dict['loss_comm'] + inv_const * loss_dict['loss_inv'] + reclat_const * loss_dict['loss_latent']
                loss.backward()
                optimizer.step()
                ppe.reporting.report({
                    'train/loss': loss.item(),
                    'train/loss_bd': loss_dict['loss_bd'].item(),
                    'train/loss_orth': loss_dict['loss_orth'].item(),
                    'train/loss_comm': loss_dict['loss_comm'].item(),
                    'train/loss_inv': loss_dict['loss_inv'].item(),
                    'train/loss_latent': loss_dict['loss_latent'].item(),
                })

            if manager.stop_trigger:
                break


def loop_simclr(manager, model, optimizer, train_loader, config, device):
    while not manager.stop_trigger:
        for images in train_loader:
            with manager.run_iteration():
                if manager.iteration >= config['training_loop']['args']['lr_decay_iter']:
                    optimizer.param_groups[0]['lr'] = config['lr']/3.
                else:
                    optimizer.param_groups[0]['lr'] = config['lr']
                model.train()
                images = torch.stack(images, dim=1).to(device)  # n t c h w
                zs = model(images)
                zs = [zs[:, i] for i in range(zs.shape[1])]
                loss = simclr(
                    zs,
                    loss_type=config['training_loop']['args']['loss_type'],
                    temperature=config['training_loop']['args']['temp']
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ppe.reporting.report({
                    'train/loss': loss.item(),
                })
            if manager.stop_trigger:
                break