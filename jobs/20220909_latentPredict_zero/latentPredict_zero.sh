python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0 reg.reg_obs=0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent0_reg_obs0


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0 reg.reg_obs=1.0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent0_reg_obs10


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 reg.reg_obs=0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent005_reg_obs0


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.05 reg.reg_obs=1.0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent005_reg_obs10


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 reg.reg_obs=0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent01_reg_obs0


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.1 reg.reg_obs=1.0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent01_reg_obs10


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.2 reg.reg_obs=0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent02_reg_obs0


python ./run.py --config_path=./configs/mnist/lstsq/neuralM.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=50000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 lr=0.0003 reg.reg_comm=None reg.reg_bd=None reg.reg_inv=None reg.reg_orth=None reg.reg_latent=0.2 reg.reg_obs=1.0 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=10 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False model.fn=./models/seqae.py model.name=SeqAENeuralM_latentPredict model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 model.args.dmode=default model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=80000 training_loop.args.reconst_iter=100000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220909_latentPredict_zero_0/reg_latent02_reg_obs10


