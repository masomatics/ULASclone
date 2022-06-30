python ./run.py --config_path=./configs/so3/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=10000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=8 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220628_so3_linear_net_0/T8


python ./run.py --config_path=./configs/so3/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=10000 report_freq=1000 model_snapshot_freq=10000 manager_snapshot_freq=1000 num_workers=2 T_cond=5 lr=0.001 train_data.fn=./datasets/so3_data.py train_data.name=SO3rotationSequence train_data.args.data_filename=so3dat_sphere_Linear.pt train_data.args.train=True train_data.args.T=10 model.fn=./models/seqae.py model.name=SeqAELSTSQ_LinearNet model.args.dim_m=10 model.args.dim_a=6 model.args.predictive=True training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/nfs-mnj-hot-01/tmp/masomatics/block_diag/result/20220628_so3_linear_net_0/T10


