python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach1_modeglasso_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach1_modeglasso_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach1_modeexact_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach1_modeexact_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach0_modeglasso_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach0_modeglasso_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach0_modeexact_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_detach0_modeexact_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach1_modeglasso_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach1_modeglasso_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach1_modeexact_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=1 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach1_modeexact_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach0_modeglasso_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=glasso model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach0_modeglasso_lr01


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.01 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach0_modeexact_lr001


python ./run.py --config_path=./configs/mnist/lstsq/lstsq_inner.yml --attr batchsize=32 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=100000 manager_snapshot_freq=100000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST_double train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=False train_data.args.pair_transition=False train_data.args.same_object=False train_data.args.fixpos=True model.fn=./models/seqae.py model.name=SeqAELSTSQ_inner model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 model.args.inner_args.normalize=0 model.args.inner_args.batchsize=32 model.args.inner_args.num_loops=20 model.args.inner_args.detach=0 model.args.inner_args.beta=0 model.args.inner_args.temperature=1 model.args.inner_args.mode=exact model.args.inner_args.lr=0.1 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20221221_glasso_mse_0/nameSequentialMNIST_double_detach0_modeexact_lr01


