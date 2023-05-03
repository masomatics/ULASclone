version=0
LOGDIR_ROOT=/mnt/vol21/masomatics/result/ulas


#Parts to change
projname=ULASclone
date=20230502
jobname=fisheyetwo
base_config=./configs/shiftim/lstsq/lstsq.yml
base_file=run.py


jobloc=./jobs/${date}_${jobname}
shellname=${jobname}
dir_path=/mnt/vol21/masomatics/${projname}
log_dir=${LOGDIR_ROOT}/$(date +'%Y%m%d')_${shellname}_${version}


python ./source/generate_minai_scripts.py --mode=raw \
--variation_path=${jobloc}/${shellname}.yaml \
--shell_path=${jobloc}/${shellname}.sh \
--log_dir=${log_dir} \
--config_path=${base_config} \
--base_file=${base_file}

python ./source/generate_minai_scripts.py --mode=minai \
--variation_path=${jobloc}/${shellname}.yaml \
--shell_path=${jobloc}/${shellname}_minai.sh \
--log_dir=${log_dir} \
--config_path=${base_config} \
--base_file=${base_file} --gpu_memory=32 \
--dir_path=${dir_path}

