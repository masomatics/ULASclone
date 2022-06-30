LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
result_dirname=temp
dataset_name=mnist
model_name=lstsq
method_name=neuralM
seed=1

python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}/ \
            --config_path=./configs/${dataset_name}/${model_name}/${method_name}.yml \
            --attr seed=${seed}