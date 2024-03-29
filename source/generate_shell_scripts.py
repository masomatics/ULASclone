import argparse
import itertools
import yaml
import functools
import copy
import pdb
import os
import numpy as np
import re
import generate_shell_helper as gsh

'''
Automated PfKube shell generator. 
It can directly use the Yaml file.
To use this file, 
(0) Decide the  base yaml config file you want to experiment on. 

(1) in your jobs folder, create a different "variation" yaml file in which
multiple values, separated by ',' , is written into 
each factor of interest (see jobs/variation.py) 

eg. 
#Basic specification
device: 0
batchsize: 20,40,50
max_epochs: 30

#Transformation specification
trf_type: discrete
trf_fxn:
  fn: data_util/kusodeka_mnist
  name: tile_all_except_one
  trf_args:
    shift_prop: 4
    patch_prop: 2,4

#Encoding network for T(x) -> Z
encoder:
  fn: model/basic_nets
  name: MLP,ConvNet0
  args:
    dim_out: 20,40
    n_layers: 2,3

head_net:
  fn: model/basic_nets
  name: MLP, ConvNet0
  args:
    dim_out: 20,40


(2) execute this script.
Upon execution, this script will...
 (a) write pfkube shellscript into the targetpath. 

 (b) write into the parent directory of the targetpath the 
 npy file that contains the array of factors you varied
 e.g. ['encoder.name' 'encoder.args.dim_out' 'encoder.args.n_layers'
 'head_net.args.dim_out']

 The written shell script contains the command for
 all variation of the config_path-designated yaml at all 
 locations specified by the variation_path-designated yaml file.
 The above example will write 3 x 2 x 2 x 2 x 2 x 2 commands. 

By changing the corescript in the main(), it can accomodate with
more variety of situations at your will.


'''













'''
variation_path :  path of the yaml file in which to look for variation

config_path : base yaml file to apply the variation 

base_file : python file to execute. 

savedir : parent directory in which to save all outputs 

shell_path : path into which to save the shellscript.

mode : if set to pfkube, it will write pfkube oriented shell.  if set to raw,
it will write locally executable shell file. raw output is useful in checking
if the code has an error.

max_gpu : referenced only when the mode is not pfkube. device value will be 
modded out by this number. 

dir_path : dir path at which the train is located.
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variation_path', type=str,
                        default="./jobs/variation.yaml")
    parser.add_argument('--config_path', type=str,
                        default="./configs/simCLR.yaml")
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('--base_file',type=str, default='train_off_the_shelf.py')
    parser.add_argument('--log_dir', type=str, default='./result/20210105')
    parser.add_argument('--shell_path',type=str, default='jobs/trial_auto.sh')
    parser.add_argument('-w', '--warning', action='store_true')
    parser.add_argument('--mode', type=str, default='pfkube')
    parser.add_argument('--max_gpu', type=int, default=8)
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--memory', type=str, default='48Gi')
    parser.add_argument('--job_version', type=str, default='0')
    parser.add_argument('--gpu_memory', type=str, default='16')
    parser.add_argument('--dir_path', type=str, default="/mnt/vol21/masomatics/ULASclone")



    args = parser.parse_args()

    with open(args.variation_path, 'r') as f:
        config = yaml.safe_load(f)

    #Modify the yaml file using attr
    for attr in args.attrs:
        module, new_value = attr.split('=')
        keys = module.split('.')
        target = functools.reduce(dict.__getitem__, keys[:-1], config)
        if keys[-1] in target.keys():
            target[keys[-1]] = yaml.safe_load(new_value)
        else:
            raise ValueError('the specified key does not exist:{}', keys)

    #create the result directory and save yaml
    variation_config = copy.deepcopy(config)
    base_config = args.config_path




    attr_list, compare_keys, dir_paths = gsh.generate_attr(variation_config, \
                              base_config, \
         log_dir=args.log_dir, mode=args.mode,
                              maxgpu=args.max_gpu)

    compare_keyfile='/'.join(((args.shell_path).split('/'))[:-1] \
                             + ['compare_keys.npy'])
    with open(compare_keyfile, 'wb') as f:
        np.save(f, compare_keys)



    #filepath to execute
    base_file_path = os.path.join(args.root_path, args.base_file)
    #write into a file : TRAINING SCRIPT
    with open(args.shell_path, "w") as shell_script:

        if args.mode == 'pfkube':
            print('kubectl delete pod --field-selector=status.phase=Succeeded', file=shell_script)
            print('kubectl get pods --field-selector "status.phase=Failed" -o name | xargs kubectl delete', file=shell_script)
            print("\n", file=shell_script)

        for k in range(len(attr_list)):
            lastcommand = ''

            if args.gpu_memory=='32':
                gpu_addendum = '-o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB"'
            else:
                gpu_addendum = ''


            if args.mode == 'pfkube':
                # corescript
                jobname = gsh.dirname_to_jobname(args.log_dir)
                jobname = jobname + str(k) +  '-' + args.job_version
                corescript1 = f"""pfkube run --job-name={jobname} \
--gpu=1 \
--persist \
--cpu={args.cpu} \
--memory={args.memory} \
--allow-overwrite \
--no-attach-logs \
{gpu_addendum} \
eval """
                corescript0 = "cd %s && "%args.dir_path
                corescript0 = corescript0 + "pip install einops && "
                corescript0 = corescript0 + "pip install opencv-python && "
                corescript0 = corescript0 + "pip install scikit-image && "
                corescript0 = corescript0 + "python %s "%base_file_path
                #corescript0 = "cd %s && python %s "%(args.dir_path, base_file_path)

                corescripts = [corescript0, corescript1]

                attr_list[k] = corescripts[0] + attr_list[k]
                attr_list[k] = '"' + attr_list[k] + lastcommand + '"'
                attr_list[k] = corescripts[1] + attr_list[k]

            elif args.mode == 'pfkube_new':
                corescript1 = "pfkube run --gpu=1 --persist --no-attach-logs eval "


            elif args.mode == 'screen':
                attr_list[k] = "python %s "%base_file_path + attr_list[k]
                attr_list[k] = "screen -S scr%s -dm "%k + attr_list[k]

            elif args.mode == 'raw':
                attr_list[k] = "python %s " %base_file_path + attr_list[k] + lastcommand

            else:
                raise NotImplementedError

            print(attr_list[k], file=shell_script)
            print("\n", file=shell_script)

