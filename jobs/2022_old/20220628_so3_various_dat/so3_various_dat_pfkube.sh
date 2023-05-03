kubectl delete pod --field-selector=status.phase=Succeeded
kubectl get pods --field-selector "status.phase=Failed" -o name | xargs kubectl delete


pfkube run --job-name=so-various-dat-zero0-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero1-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero2-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero3-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero4-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero5-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.01 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero6-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero7-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero8-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero9-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero10-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero11-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr0001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero12-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero13-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_Linearpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero14-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero15-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_iResNet.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_iResNetpt_nameSeqAELSTSQ_LinearNet"


pfkube run --job-name=so-various-dat-zero16-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_so3Net model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_so3Net"


pfkube run --job-name=so-various-dat-zero17-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/so3/lstsq/lstsq_resnet.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.0001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_MLP.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.ch_x=1 model.args.k=2.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220630_so3_various_dat_0/lr00001_data_filenameso3dat_sphere_MLPpt_nameSeqAELSTSQ_LinearNet"

