LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
result_dirname=20220615_default_run_mnist
dataset_name=mnist_bg
model_name=lstsq
seed=1

python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}/ \
            --config_path=./configs/${dataset_name}/${model_name}/${model_name}.yml \
            --attr seed=${seed}



