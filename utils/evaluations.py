import torch
import pdb
def predict(images, model,
            n_cond=2, tp=5, device='cpu', swap =False,
            predictive=False):

    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)

        # pdb.set_trace()
    images = images.to(device)
    images_cond = images[:, :n_cond]

    M = model.get_M(images_cond)  # n a a
    if type(M) == tuple:
        M = M[0]

    if predictive:
        H = model.encode(images_cond[:, [0]])[:, 0]
        tp = n_cond -1 + tp
        xs0 = images[:, [0]].to('cpu')
    else:
        H = model.encode(images_cond[:, -1:])[:, 0] # n s a
        xs0 = []
    xs = []
    n, s, a = H.shape


    if swap == True:
        M = M[torch.arange(-n//2, n-n//2)]

    for r in range(tp):
        H = H @ M
        x_next_t = model.decode(H[:, None])
        xs.append(x_next_t)
    x_next = torch.sigmoid(torch.cat(xs, axis=1).detach().to('cpu'))
    if len(xs0) > 0:
        x_next = torch.cat([xs0] +[x_next], axis=1)

    return x_next, M