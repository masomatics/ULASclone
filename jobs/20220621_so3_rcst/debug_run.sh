LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
result_dirname=temp
dataset_name=mnist
model_name=lstsq
seed=1

python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}/ \
            --config_path=./configs/${dataset_name}/${model_name}/${model_name}.yml \
            --attr seed=${seed}



LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
seed=1
result_dirname=20220621_so3

method_name=so3run
python run.py --log_dir=${LOGDIR_ROOT}/temp/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}
