import os
import numpy as np
import yaml
import torch


def iter_list(log_dir):
    dirlist = os.listdir(log_dir)
    iterlist = []
    for mydir in dirlist:
        if 'iter' in mydir and 'tmp' not in mydir:
            iterlist.append(int(mydir.split('_')[-1]))
    return np.array(iterlist)


def load_model(model, log_dir, iters, latest=False):
    print(f"""{log_dir} \n snapshot_model_iter_{iters}""")
    if latest == True:
        iters = np.max(iter_list(log_dir))
    model.load_state_dict(torch.load
        (os.path.join(log_dir, 'snapshot_model_iter_{}'.format(iters))))



def load_config(targdir_path):
    targ_config_path = os.path.join(targdir_path, 'config.yml')
    with open(targ_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def filter_names(query, mydict):
    filtered = {}
    for key in mydict.keys():
        if query in key:
            filtered[key] = mydict[key]
    return filtered

def filter_list(query, mylist):
    filtered = []
    for key in mylist:
        if query in key:
            filtered.append(key)
    return filtered


def model_exists(targlist,  targpath):
    filtered_list = []
    for targdir in targlist:
        if os.path.exists(os.path.join(targpath, targdir, 'config.yml')):
            filtered_list.append(targdir)
    return filtered_list



def read_log(targpath):
    logpath = os.path.join(targpath, 'log')
    if os.path.exists(logpath):
        with open(logpath) as f:
            log = yaml.safe_load(f)
        return log
    else:
        return 0


def read_history(targpath):
    log = read_log(targpath)

    dict_history = {}
    if log ==0:
        pass
    else:
        querykeys = list(log[0].keys())
        for key in querykeys:
            values = []
            for k in range(len(log)):
                values.append(log[k][key])
            dict_history[key] = values
    return dict_history

