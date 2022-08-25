import torch
def predict(images, model,
            n_cond=2, tp=5, device='cpu'):

    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)

        # pdb.set_trace()
    images = images.to(device)
    images_cond = images[:, :n_cond]
    images_target = images[:, n_cond:n_cond + tp]

    M = model.get_M(images_cond)  # n a a
    H = model.encode(images_cond[:, -1:])[:, 0]  # n s a
    xs = []
    for r in range(tp):
        H = H @ M
        x_next_t = model.decode(H[:, None])
        xs.append(x_next_t)

    x_next = torch.sigmoid(torch.cat(xs, axis=1).detach().to('cpu'))

    return x_next, M