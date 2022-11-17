version=0
LOGDIR_ROOT=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result


#Parts to change
projname=ULASclone
date=20221117
jobname=mnist_double_two
base_config=./configs/mnist/lstsq/lstsq.yml
base_file=run.py


jobloc=./jobs/${date}_${jobname}
shellname=${jobname}
dir_path=/mnt/vol21/masomatics/${projname}
log_dir=${LOGDIR_ROOT}/$(date +'%Y%m%d')_${shellname}_${version}


python ./source/generate_shell_scripts.py --mode=raw \
--variation_path=${jobloc}/${shellname}.yaml \
--shell_path=${jobloc}/${shellname}.sh \
--log_dir=${log_dir} \
--config_path=${base_config} \
--base_file=${base_file}

python ./source/generate_shell_scripts.py --mode=pfkube \
--variation_path=${jobloc}/${shellname}.yaml \
--shell_path=${jobloc}/${shellname}_pfkube.sh \
--log_dir=${log_dir} \
--config_path=${base_config} \
--base_file=${base_file} --gpu_memory=32 \
--dir_path=${dir_path}

