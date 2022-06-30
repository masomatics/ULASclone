python ./run.py --config_path=./configs/mnist/lstsq/lstsq.yml --attr batchsize=16 seed=1 max_iteration=100000 report_freq=1000 model_snapshot_freq=25000 manager_snapshot_freq=50000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 train_data.fn=./datasets/seq_mnist.py train_data.name=SequentialMNIST train_data.args.root=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/datasets/MNIST train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_angle_velocity_ratio=[-0.5,0.5] train_data.args.max_color_velocity_ratio=[-0.5,0.5] train_data.args.only_use_digit4=True train_data.args.backgrnd=True model.fn=./models/seqae.py model.name=SeqAENeuralM_comm model.args.dim_m=256 model.args.dim_a=16 model.args.ch_x=3 model.args.k=4.0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=50000 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/temp
