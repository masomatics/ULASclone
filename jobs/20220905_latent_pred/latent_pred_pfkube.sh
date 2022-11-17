kubectl delete pod --field-selector=status.phase=Succeeded
kubectl get pods --field-selector "status.phase=Failed" -o name | xargs kubectl delete


pfkube run --job-name=latent-pred-zero0-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent005_T3_predictiveTrue"


pfkube run --job-name=latent-pred-zero1-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent005_T3_predictiveFalse"


pfkube run --job-name=latent-pred-zero2-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent005_T5_predictiveTrue"


pfkube run --job-name=latent-pred-zero3-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent005_T5_predictiveFalse"


pfkube run --job-name=latent-pred-zero4-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent01_T3_predictiveTrue"


pfkube run --job-name=latent-pred-zero5-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent01_T3_predictiveFalse"


pfkube run --job-name=latent-pred-zero6-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent01_T5_predictiveTrue"


pfkube run --job-name=latent-pred-zero7-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent01_T5_predictiveFalse"


pfkube run --job-name=latent-pred-zero8-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=1. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent1_T3_predictiveTrue"


pfkube run --job-name=latent-pred-zero9-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=1. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent1_T3_predictiveFalse"


pfkube run --job-name=latent-pred-zero10-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=1. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent1_T5_predictiveTrue"


pfkube run --job-name=latent-pred-zero11-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=1. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent1_T5_predictiveFalse"


pfkube run --job-name=latent-pred-zero12-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=2. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent2_T3_predictiveTrue"


pfkube run --job-name=latent-pred-zero13-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=2. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent2_T3_predictiveFalse"


pfkube run --job-name=latent-pred-zero14-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=2. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent2_T5_predictiveTrue"


pfkube run --job-name=latent-pred-zero15-0 --gpu=1 --persist --cpu=8 --memory=48Gi --allow-overwrite --no-attach-logs -o nodeSelector="nvidia.k8s.pfn.io/gpu_model: Tesla-V100-SXM2-32GB" eval "cd /mnt/vol21/masomatics/ULASclone && pip install einops && pip install opencv-python && pip install scikit-image && python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=2. train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=5 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.predictive=False training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220906_latent_pred_0/reg_latent2_T5_predictiveFalse"


