import torch
import torch.nn as nn
from einops import repeat, rearrange
import pdb
from utils import optimize_bd_cob as obc
import copy
from source import yaml_utils as yu
from utils import notebook_utils as nu
from datasets import seq_mnist as sm

class ChangeOfBasisR(torch.nn.Module):
    def __init__(self, d, Pmat=torch.tensor([]), reg=0.0):
        super().__init__()
        if len(Pmat) > 0:
            self.U = nn.Parameter(Pmat)
        else:
            self.U = nn.Parameter(torch.empty(d, d))
        torch.nn.init.orthogonal_(self.U)
        self.reg = reg
        self.dim = d

    def __call__(self, encoded):
        if len(encoded.shape) == 4:
            encoded_hat = torch.einsum('b t s a, s r ->b t r a', encoded, self.U)
        else:
            encoded_hat = torch.einsum('t s a, s r -> t r a', encoded, self.U)
        return encoded_hat

    def grouplasso_loss(self, z, z0):

        delta = self(z- z0)

        if len(delta.shape) == 4:
            delta = rearrange(delta, 'b t s a -> (b t a) s')
        else:
            delta = rearrange(delta, 't s a -> (t a) s')
        grplasso_loss = torch.sum(torch.sqrt(torch.sum(delta ** 2, axis=0)))

        grplasso_loss = grplasso_loss + self.reg * self.normreg()
        #grplasso_loss = grplasso_loss + self.reg * self.orthreg()

        return grplasso_loss, delta.detach()

    def normreg(self):
        regval =torch.sum(torch.abs(1 - torch.sqrt(torch.sum(self.U**2, axis=1))))
        return regval

    def orthreg(self):
        orthdelta = self.U @ self.U - torch.eye(self.dim).to(self.U.device)
        regval = torch.sum(orthdelta**2)
        return regval


    def normalize_vec(self):
        self.U = nn.Parameter(self.U / torch.sqrt(torch.sum(self.U ** 2, axis=0, keepdims=True)))

    def undo(self, encoded):
        invmat = torch.linalg.inv(self.U)
        if len(encoded.shape) == 4:
            encoded_hat = torch.einsum('b t s a, s r ->b t r a', encoded, invmat)
        else:
            encoded_hat = torch.einsum('t s a, s r -> t r a', encoded, invmat)
        return encoded_hat

def sort_pair_col(tensor1, tensor2):
    one_wins = []
    two_wins = []
    for j in range(tensor1.shape[-1]):
        if torch.sum(tensor1[:,j]**2) > torch.sum(tensor2[:, j]**2):
            one_wins.append(j)
        else:
            two_wins.append(j)
    wholelist = one_wins + two_wins
    return tensor1[:, wholelist], tensor2[:, wholelist], (one_wins, two_wins)


def apply_to_select_col(M, tensor, cols):
    tensor = copy.deepcopy(tensor)
    tensor[cols] = tensor[cols] @ M
    return tensor


def apply_to_selectcol_list(M, tensor, blockposlist, collist):
    transformed = copy.deepcopy(tensor)
    for k in range(len(blockposlist)):
        selection = blockposlist[k]
        cols = collist[k]
        Mblock = M[:, selection][selection, :]
        transformed[:, selection] = apply_to_select_col(Mblock, tensor[:, selection], cols)
    return transformed


def covariance(H):
    if len(H.shape) == 4:
        EH2 = torch.einsum('b t s a, b t r a -> b t s r', H, H)
        EH2 = torch.mean(EH2, axis=[0,1])

        EH = torch.mean(H, axis=[0,1])
        EHEH = torch.einsum('s a, r a -> s r', EH, EH)
        pass

    else:
        EH2 = torch.einsum('t s a, t r a -> t s r', H, H)
        EH2 = torch.mean(EH2, axis=0)

        EH = torch.mean(H, axis = 0)
        EHEH = torch.einsum('s a, r a -> s r', EH, EH)
    covariance = EH2 - EHEH
    return covariance


def innerprod(H):
    if len(H.shape) == 4:
        EH2 = torch.einsum('b t s a, b t r a -> b t s r', H, H)
        EH2 = torch.mean(EH2, axis=[0,1])
        pass

    else:
        EH2 = torch.einsum('t s a, t r a -> t s r', H, H)
        EH2 = torch.mean(EH2, axis=0)

    prod = EH2
    return prod


def optimize_cov_blocks(H,
                    n_epochs=2000,
                    epochs_monitor=500,
                    verbose=False,
                    lr=0.02, normalize=True):
    # Optimize change of basis matrix U by minimizing block diagonalization loss
    #Change of basis R is to be applied to  b t s a shape for t s a shape, acting on s dimension
    change_of_basis = ChangeOfBasisR(d=H.shape[-2], reg=0.0).to(H.device)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=lr)
    loss_rec = []
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        HP = change_of_basis(H)
        covHP = torch.abs(covariance(HP))
        loss = torch.mean(obc.tracenorm_of_normalized_laplacian(covHP))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if ((ep+1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss))
        loss_rec.append(total_loss)
    if normalize == True:
        change_of_basis.normalize_vec()
    if verbose:
        return change_of_basis, torch.tensor(loss_rec)
    else:
        return change_of_basis

def obtain_pair_sequence(checkmodelpath, check_idx=0, T=30):
    config = nu.load_config(checkmodelpath)
    data_args = config['train_data']['args']
    data_args['T'] = T
    double_dat = sm.SequentialMNIST_double(**data_args)
    datseq = double_dat.one_obj_immobile(check_idx)
    datseq_two = double_dat.one_obj_immobile(check_idx, mode=0)
    return datseq, datseq_two



def trainP(seqslice, seqslice2, iters=400, lr=0.01, monitor=50, reg=0.1):
    change_of_basisR = ChangeOfBasisR(d=seqslice.shape[-2], reg=reg).to(seqslice.device)
    optimizer = torch.optim.Adam(change_of_basisR.parameters(), lr=lr)
    for k in range(iters):
        loss1, _  = change_of_basisR.grouplasso_loss(seqslice[1:], seqslice[0])
        loss2, _ = change_of_basisR.grouplasso_loss(seqslice2[1:], seqslice2[0])
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (k % monitor) == 0:
            print(f"""iter {k} : loss {loss}""")
    return change_of_basisR


    # config = nu.load_config(checkmodelpath)
    # datseq, datseq_two = obtain_pair_sequence(checkmodelpath, check_idx=check_idx, T=T)
    # model_config = config['model']
    # checkmodel = yu.load_component(model_config)
    # encoded = checkmodel.encode(torch.stack(datseq).unsqueeze(0))[0].detach()
    # encoded_two = checkmodel.encode(torch.stack(datseq_two).unsqueeze(0))[0].detach()


'''
FIND THE Isotypic Dimension Mixer over multiple ORBITS

'''
def optimize_cov_blocks_inter(config, change_basisL,
                    n_epochs=30,
                    epochs_monitor=1,
                    verbose=False,
                    lr=0.02, normalize=True,
                              tp=10,
                              n_cond=2,
                              device=0,
                              selection_dim=[0]):
    dataconfig = config['train_data']
    dataconfig['args']['T'] = tp + n_cond
    if dataconfig['args']['pair_transition'] != True:
        dataconfig['args']['train'] = False
    dataconfig['args']['max_T'] = tp + n_cond

    data = yu.load_component(dataconfig)
    train_loader = torch.utils.data.DataLoader(data,
                              batch_size=config['batchsize'],
                              shuffle=True,
                              num_workers=config['num_workers'])


    # Optimize change of basis matrix U by minimizing block diagonalization loss
    dataseq = torch.stack(iter(train_loader).next())
    dataseq = dataseq.to(device)

    model_config = config['model']
    model = yu.load_component(model_config)
    model = model.to(device)
    H = model.encode(dataseq)

    #This is the one to be trained
    change_of_basis = ChangeOfBasisR(d=H.shape[-2], reg=0.0).to(H.device)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=lr)
    change_basisL = change_basisL.to(device)
    loss_rec = []
    print(len(train_loader))
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        for dataseq in train_loader:
            dataseq = torch.stack(dataseq).to(device)
            H = model.encode(dataseq).detach()
            #Fourier transform
            H = (H @ change_basisL.U).detach()[:,:, :, selection_dim]
            HP = change_of_basis(H)
            covHP = torch.abs(covariance(HP))
            loss = torch.mean(obc.tracenorm_of_normalized_laplacian(covHP))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss = total_loss/len(train_loader)

        if ((ep+1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss))
        loss_rec.append(total_loss)
        if normalize == True:
            change_of_basis.normalize_vec()
    if verbose:
        return change_of_basis, torch.tensor(loss_rec), covHP
    else:
        return change_of_basis



def move_partial_onestep(selections, H, M, partitions, part_idx,
                         dim1treatment=False):
    # H : s a
    # M : a a
    Hcopy = copy.deepcopy(H.detach())

    for k in range(len(selections)):
        selection = selections[k]

        if dim1treatment == True and len(selection) == 1:
            part= list(range(Hcopy.shape[0]))
        else:
            partition = partitions[k]
            #extract the part
            part = partition[part_idx]
        Hpart = Hcopy[part]    #H: part, a

        #isolate the frequency
        H_select = Hpart[:, selection]    #H : part, select
        M_select = M[selection, :][:, selection]   #M : select, select

        #apply the difference in frequency
        H_alter = H_select @ M_select     #H : part, select

        Hcopy = assign(Hcopy, H_alter, part, selection)

    return Hcopy.detach()


'''
VERY USEFUL HELPER FUNCTION that assigns matrix to a specific position of the 
mothermatrix. mat[[rows]][: cols] = submat DOES NOT WORK because of the 
construct of pytorch.
'''
def assign(mat, submat, rows, cols):
    for k in range(len(submat)):
        mat[rows[k]][cols]  = submat[k]
    return mat


def move_partial(selections, H, M, partitions, part_idx, T, dim1treatment=False):
    Hs = [H]
    H_old = H
    for k in range(T):
        Hnew = move_partial_onestep(selections, H_old, M, partitions, part_idx,
                                    dim1treatment=dim1treatment)
        Hs.append(Hnew)
        H_old = Hnew

    Hs = torch.stack(Hs)
    return Hs


