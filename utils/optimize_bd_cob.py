import torch
import torch.nn as nn
from einops import repeat
from utils.laplacian import tracenorm_of_normalized_laplacian, make_identity_like
import numpy as np
import pdb
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

class ChangeOfBasis(torch.nn.Module):
    def __init__(self, d, Pmat=torch.tensor([])):
        super().__init__()
        if len(Pmat) > 0:
            self.U = nn.Parameter(Pmat)
        else:
            self.U = nn.Parameter(torch.empty(d, d))
        torch.nn.init.orthogonal_(self.U)

    def __call__(self, mat):
        _U = repeat(self.U, "a1 a2 -> n a1 a2", n=mat.shape[0])
        n_mat = torch.linalg.solve(_U, mat) @ _U
        return n_mat

    def apply_to_tensor(self, tensor):
        return tensor @ self.U

    def normalize_vec(self):
        self.U = nn.Parameter(self.U / torch.sqrt(torch.sum(self.U ** 2, axis=0, keepdims=True)))


def optimize_bd_cob(mats,
                    batchsize=32,
                    n_epochs=50,
                    epochs_monitor=10,
                    verbose=False,
                    lr=0.1,
                    normalize=True):
    # Optimize change of basis matrix U by minimizing block diagonalization loss

    change_of_basis = ChangeOfBasis(mats.shape[-1]).to(mats.device)
    dataloader = torch.utils.data.DataLoader(
        mats, batch_size=batchsize, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(change_of_basis.parameters(), lr=lr)
    loss_rec = []
    for ep in range(n_epochs):
        total_loss, total_N = 0, 0
        for mat in dataloader:
            n_mat = change_of_basis(mat)
            n_mat  = torch.abs(n_mat)
            n_mat = torch.matmul(n_mat.transpose(-2, -1), n_mat)
            #n_mat = torch.abs(n_mat) + torch.abs(n_mat.transpose(-2, -1))
            loss = torch.mean(
                tracenorm_of_normalized_laplacian(torch.abs(n_mat)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mat.shape[0]
            total_N += mat.shape[0]
        if ((ep+1) % epochs_monitor) == 0:
            print('ep:{} loss:{}'.format(ep, total_loss/total_N))
        loss_rec.append(total_loss/total_N)

    if normalize == True:
        change_of_basis.normalize_vec()
    if verbose:
        return change_of_basis, torch.tensor(loss_rec)
    else:
        return change_of_basis


def obtain_blocks_old(M):
    alldims = list(range(M.shape[1]))
    partitions = []
    def removelist(mylist, toberemoved):
        for j in toberemoved:
            if j in mylist:
                mylist.remove(j)
        return mylist

    taken = []
    while len(alldims) > 0:
        idx = alldims[0]
        deviate = M[idx] - torch.mean(M[idx])
        thresh = torch.std(deviate)

        preblock = list(np.where(deviate > thresh)[0])
        partitions = mergelist(partitions, preblock)
        # print(partitions)
        # pdb.set_trace()

        # block = partitions[-1]
        #
        # removelist(alldims, block)
        # if preblock == block or len(preblock) == 0:
        #     partitions = mergelist(partitions, [idx])
        #     removelist(alldims, [idx])
        #
        # taken = taken + partitions[-1]


        block = list(np.where(deviate > thresh)[0])
        removelist(preblock, taken)
        removelist(alldims, block)
        taken = taken + block
        # if len(block) ==0:
        #     removelist(alldims, [idx])
        # else:
        #     partitions.append(block)


        #print(len(alldims), len(block), len(taken))

    return partitions


def obtain_blocks(M, threshconst=1.0):
    alldims = list(range(M.shape[1]))
    partitions = []
    def removelist(mylist, toberemoved):
        for j in toberemoved:
            if j in mylist:
                mylist.remove(j)
        return mylist

    taken = []
    while len(alldims) > 0:
        #idx = alldims[np.random.choice(len(alldims))]
        idx = alldims[0]
        deviate = M[idx] - torch.mean(M[idx])
        thresh = threshconst * torch.std(deviate)
        preblock = list(np.where(deviate > thresh)[0])
        partitions = mergelist(partitions, preblock)
        block = partitions[-1]

        oldlen = len(alldims)
        removelist(alldims, block)
        if oldlen == len(alldims):
            alldims.remove(idx)

        taken = list(set(taken + block))

        if len(alldims) == 103:
            pdb.set_trace()

        print(len(alldims), len(block), len(taken))
    return partitions


def blockify(checkmat, thresh=1., hard =True):

    if hard == True:
        adjmat = np.zeros(checkmat.shape)
        for k in range(len(adjmat)):
            adjvec = checkmat[k].numpy()
            adjvec = np.array(adjvec > np.mean(adjvec) + thresh*np.std(adjvec))
            adjmat[k] = adjvec
    else:
        adjmat = checkmat

    graph = csr_matrix(adjmat)
    perm = list(reverse_cuthill_mckee(graph))
    return checkmat[perm,:][:, perm], perm




"""
merge all blocks that intersect at block A
"""
def mergelist(blocks, blockA):
    merged = blockA
    toremove = []
    for block in blocks:
        if len(set(block).intersection(set(blockA))) > 0:
            merged = list(set(merged + block))
            toremove = toremove + [block]
    blocks = blocks + [merged]
    for block in toremove:
        blocks.remove(block)

    return blocks

