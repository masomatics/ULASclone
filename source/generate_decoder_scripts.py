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


if __name__ == '__main__':
    rootpath = '/mnt/nfs-mnj-hot-01/tmp/masomatics/Symmetry_Adaptation/result'

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir_name', type=str)
    parser.add_argument('--root', type=str, default =rootpath)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--shell_path',type=str, default='jobs/trial_auto.sh')
    parser.add_argument('--mode', type=str, default='pfkube')
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--memory', type=str, default='48Gi')
    parser.add_argument('--jobname', type=str, default='trial')

    args = parser.parse_args()


    folder_path = os.path.join(args.root, args.eval_dir_name)

    #For VANILLA, we do not look through the encoder folders
    if args.eval_dir_name == 'vanilla':
        folder_list =['']
    else:
        folder_list = os.listdir(folder_path)


    #write into a file : Evaluation SCRIPT
    with open(args.shell_path, "w") as eval_script:

        if args.mode == 'pfkube':
            print('kubectl delete pod --field-selector=status.phase=Succeeded', file=eval_script)
            print('kubectl get pods --field-selector "status.phase=Failed" -o name | xargs kubectl delete', file=eval_script)
            print("\n", file=eval_script)

        for k in range(len(folder_list)):

            #vanilla case treatment
            if len(folder_list[k]) > 0:
                targdir = os.path.join(args.eval_dir_name, folder_list[k])
            else:
                targdir = args.eval_dir_name
            base_eval_command = f"""python ./decoder_train.py \
--targdir={targdir} \
--num_epochs={args.num_epochs} \
--lr={args.lr} \
--batchsize={args.batchsize}"""


            if args.mode == 'pfkube':
                jobname = gsh.dirname_to_jobname_ex(args.jobname)
                jobname = jobname + str(k) + '-eval'
                corescript1 = f"""pfkube run --job-name={jobname} \
--gpu=1 --cpu={args.cpu} \
--memory={args.memory} \
--persist \
--allow-overwrite \
--no-attach-logs eval """
                corescript0 = "cd /mnt/vol21/masomatics/Symmetry_Adaptation && "
                corescript0 = corescript0 + "pip install opencv-python && "
                corescript0 = corescript0 + "pip install scikit-image && "
                corescripts = [corescript0, corescript1]

                line_for_dir_idxk = corescript0 + base_eval_command
                line_for_dir_idxk = '"' + line_for_dir_idxk + '"'
                line_for_dir_idxk = corescript1 + line_for_dir_idxk


            elif args.mode == 'screen':
                line_for_dir_idxk = base_eval_command
                line_for_dir_idxk = "screen -S scr%s -dm " % k + line_for_dir_idxk

            elif args.mode == 'raw':
                line_for_dir_idxk = base_eval_command

            else:
                raise NotImplementedError

            print(line_for_dir_idxk , file=eval_script)
            print("\n", file=eval_script)



