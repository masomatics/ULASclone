LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result
dataset_name=mnist
seed=1
result_dirname=20220615_NeuralMstar

method_name=neuralM
python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}

method_name=neuralM_vanilla
python run.py --log_dir=${LOGDIR_ROOT}/${result_dirname}_${method_name}/ \
            --config_path=./jobs/${result_dirname}/${method_name}.yml \
            --attr seed=${seed}