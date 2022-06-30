#LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
#result_dirname=20220611_so3
#dataset_name=so3
#model_name=lstsq
#seed=1
#
#python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}/ \
#            --config_path=./configs/${dataset_name}/${model_name}/${model_name}.yml \
#            --attr seed=${seed}
#


LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
seed=1
result_dirname=20220621_so3_rcst

method_name=so3run_rcst
python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}


method_name=so3run
python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}



