import torch
import pdb
from torch.utils.data import DataLoader
from utils import notebook_utils as nu
from source import yaml_utils as yu
import numpy as np
import os
from tqdm import tqdm



'''
Snippets from the functions used in the notebooks

'''

result_dir = '/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result'
baseline_path = os.path.join(result_dir, '20220615_default_run_mnist')
basestar_path = os.path.join(result_dir, '20220615_NeuralMstar_neuralM_vanilla')


#might be overlapping with get_predict
def predict(images, model,
            n_cond=2, tp=5, device='cpu', swap =False,
            predictive=False, reconstructive=False):

    if type(images) == list:
        images = torch.stack(images)
        images = images.transpose(1, 0)

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
        if reconstructive:
            xs0 = torch.sigmoid(model.decode(model.encode(images_cond[:, :n_cond])).detach().to('cpu'))
    xs = []
    n, s, a = H.shape


    if swap == True:
        M = M[torch.arange(-n//2, n-n//2)]

    for r in range(tp):
        H = H @ M[:H.shape[0]]
        x_next_t = model.decode(H[:, None])
        xs.append(x_next_t)
    x_next = torch.sigmoid(torch.cat(xs, axis=1).detach().to('cpu'))
    if len(xs0) > 0:
        x_next = torch.cat([xs0] +[x_next], axis=1)

    return x_next, M




def prediction_evaluation(targdir_pathlist, device =0,
                           n_cond=2, tp=1, repeats=3,
                           predictive=False,reconstructive = False,
                          alteration={}, 
                          mode='default'):
    results = {}
    inferred_Ms = {}
    model_configs = {}
    models = {}
    all_configs = {}

    for targdir_path in targdir_pathlist:

        if os.path.exists(os.path.join(targdir_path, 'config.yml')):
            config = nu.load_config(targdir_path)
        else:
            config = nu.load_config(baseline_path)

        config = yu.alter_config(config, alteration)

        dataconfig = config['train_data']
        dataconfig['args']['T'] = tp + n_cond
        try:
            # IF double dataset, we work with same objects
            if dataconfig['name'] == 'SequentialMNIST_double':
                dataconfig['args']['train'] = True
            else:
                dataconfig['args']['train'] = False
        except:
            print("Not working with the pair dataset")
        dataconfig['args']['max_T'] = tp + n_cond


        data = yu.load_component(dataconfig)

        train_loader = DataLoader(data,
                                  batch_size=config['batchsize'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])
        print(dataconfig)

        model_config = config['model']
        model = yu.load_component(model_config)
        iterlist = nu.iter_list(targdir_path)



        if len(iterlist) == 0:
            print(f"""There is no model trained for {targdir_path}""")
        else:
            maxiter = np.max(nu.iter_list(targdir_path))

            try:
                nu.load_model(model, targdir_path, maxiter)
            except:
                pdb.set_trace()
            model = model.eval().to(device)

            with torch.no_grad():
                l2scores = []
                for j in range(repeats):
                    Mlist = []
                    for images in tqdm(train_loader):
                        if type(images) == list:
                            images = torch.stack(images)
                            images = images.transpose(1, 0)
                        # n t c w h
                        images = images.to(device)

                        if mode == 'notebook':
                            images = images.permute([0, 1, -1, 2, 3])

                        if predictive == True or reconstructive == True:
                            images_target = images
                        else:
                            images_target = images[:, n_cond:n_cond + tp]
                        x_next, M = predict(images, model, n_cond=n_cond,
                                               tp=tp, device=device,
                                               predictive=predictive,
                                               reconstructive=reconstructive)
                        l2_losses = torch.sum(
                            (images_target.to('cpu') - x_next.to('cpu')) ** 2,
                            axis=[-1, -2, -3])
                        l2scores.append(l2_losses)

                        Mlist.append(M.detach().to('cpu'))

                    Mlist = torch.cat(Mlist)


            l2scores = torch.cat(l2scores)
            av_l2 = torch.mean(l2scores, axis=0)
            av_l2var = torch.std(l2scores, axis=0)
            print(av_l2)
            results[targdir_path] = [av_l2, av_l2var]

            inferred_Ms[targdir_path] = Mlist
            models[targdir_path] = model.to('cpu')
            model_configs[targdir_path] = model_config
            all_configs[targdir_path] = config

    output={'results':results,
            'Ms': inferred_Ms,
            'configs': all_configs,
            'models': models}

    return output, images_target.to('cpu'), x_next.to('cpu')


def get_predict(images, targdir_path, swap=False, predictive=False,device=0,
                n_cond=2, tp=1):
    if os.path.exists(os.path.join(targdir_path, 'config.yml')):
        config = nu.load_config(targdir_path)
    else:
        config = nu.load_config(baseline_path)

    # config = load_config(targdir_path)

    model_config = config['model']
    if len(nu.iter_list(targdir_path)) > 0:
        maxiter = np.max(nu.iter_list(targdir_path))
        model = yu.load_component(model_config).to(device)
        nu.load_model(model, targdir_path, maxiter)
        model = model.eval().to(device)
        # model(images[:, :2])
        if str(type(model)).split(' ')[-1].split('.')[-1].split("'")[
            0] == 'SeqAENeuralM_latentPredict':
            model.conduct_prediction(images[:, :n_cond], n_rolls=tp)
        else:
            model(images[:, :n_cond])
        x_next, M = predict(images, model, n_cond=n_cond, tp=tp,
                               device=device, swap=swap,
                               predictive=predictive)
        return x_next, M
    else:
        return 0, 0

'''
####################################################################################
###########
###########
###########
###########EQUIV EVALUATIONS
###########
###########
###########
###########
##################################################################
'''

def equiv_evalutation(targdir_pathlist, device =0,
                           n_cond=2, tp=1, repeats=3):
    equiv_results = {}
    inferred_Ms = {}
    for targdir_path in targdir_pathlist:

        if os.path.exists(os.path.join(targdir_path, 'config.yml')):
            config = nu.load_config(targdir_path)
        else:
            config = nu.load_config(baseline_path)

        dataconfig = config['train_data']
        dataconfig['args']['train'] = False
        dataconfig['args']['shared_transition'] = 1
        dataconfig['args']['T'] = tp + n_cond

        data = yu.load_component(dataconfig)
        train_loader = DataLoader(data,
                                  batch_size=config['batchsize'],
                                  shuffle=True,
                                  num_workers=config['num_workers'])
        model = yu.load_component(config['model'])
        iterlist = nu.iter_list(targdir_path)

        if len(iterlist) == 0:
            print(f"""There is no model trained for {targdir_path}""")
        else:
            maxiter = np.max(nu.iter_list(targdir_path))
            nu.load_model(model, targdir_path, maxiter)
            model = model.eval().to(device)

            # Initialize lazy modules
            images = iter(train_loader).next()
            # Initialize lazy modules
            if type(images) == list:
                images = torch.stack(images)
                images = images.transpose(1, 0)

            images = images.to(device)

            if str(type(model)).split(' ')[-1].split('.')[-1].split("'")[
                0] == 'SeqAENeuralM_latentPredict':
                model.conduct_prediction(images[:, :n_cond], n_rolls=tp)
            else:
                model(images[:, :n_cond])

            with torch.no_grad():
                l2scores = []

                for j in range(repeats):
                    Mlist = []
                    for images in tqdm(train_loader):

                        if type(images) == list:
                            images = torch.stack(images)
                            images = images.transpose(1, 0)
                            # n t c w h
                        images = images.to(device)
                        images_target = images[:, n_cond:n_cond + tp]

                        x_next_perm, M = predict(images, model,
                                                    n_cond=n_cond, tp=tp,
                                                    device=device, swap=True)

                        l2_losses = torch.sum((images_target.to(
                            'cpu') - x_next_perm.to('cpu')) ** 2,
                                              axis=[-1, -2, -3])
                        l2scores.append(l2_losses)

                        Mlist.append(M)

                        train_loader.dataset.init_shared_transition_parameters()

            Mlist = torch.cat(Mlist)
            scores = torch.cat(l2scores)
            print(l2_losses.shape)
            print(scores.shape)
            av_score = torch.mean(scores, axis=0)
            av_std = torch.std(scores, axis=0)
            print(f"""mean: {av_score}""")
            print(f"""std: {av_std}""")
            equiv_results[targdir_path] = [av_score, av_std]
            inferred_Ms[targdir_path] = Mlist
    output ={'equiv_results': equiv_results,
             'Ms' : inferred_Ms}

    return output


