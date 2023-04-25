LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
dataset_name=mnist_bg
seed=1
result_dirname=20220624_Mstar_longer_tp

method_name=neuralM_comm
python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}

#method_name=neuralM_vanilla
#python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
#            --config_path=./jobs/${result_dirname}/${method_name}.yml \
#            --attr seed=${seed}


#LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
#result_dirname=20220621_default_run_mnist_bg
#dataset_name=mnist_bg
#model_name=lstsq
#seed=1
#
#python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}/ \
#            --config_path=./configs/${dataset_name}/${model_name}/${model_name}.yml \
#            --attr seed=${seed}