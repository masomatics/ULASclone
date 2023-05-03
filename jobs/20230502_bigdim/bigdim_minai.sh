python new_minai_workflow.py --option "--config_path=./configs/shiftim/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=1000 report_freq=10 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=8 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/shift_img.py train_data.name=Shift_cifar train_data.args.root=/mnt/vol21/hayasick/data/ train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_vshift=[-5,5] train_data.args.max_hshift=[-5,5] train_data.args.deform=True model.fn=./models/seqae.py model.name=SeqAELSTSQ model.args.dim_m=512 model.args.dim_a=8 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/vol21/masomatics/result/ulas/temp" | kubectl create -f -


python new_minai_workflow.py --option "--config_path=./configs/shiftim/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=10 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=8 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/shift_img.py train_data.name=Shift_cifar train_data.args.root=/mnt/vol21/hayasick/data/ train_data.args.train=True train_data.args.T=3 train_data.args.max_T=9 train_data.args.max_vshift=[-5,5] train_data.args.max_hshift=[-5,5] train_data.args.deform=True model.fn=./models/seqae.py model.name=SeqAELSTSQ model.args.dim_m=512 model.args.dim_a=4 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/vol21/masomatics/result/ulas/20230502_bigdim_0/T3_dim_a4" | kubectl create -f -


python new_minai_workflow.py --option "--config_path=./configs/shiftim/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=10 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=8 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/shift_img.py train_data.name=Shift_cifar train_data.args.root=/mnt/vol21/hayasick/data/ train_data.args.train=True train_data.args.T=4 train_data.args.max_T=9 train_data.args.max_vshift=[-5,5] train_data.args.max_hshift=[-5,5] train_data.args.deform=True model.fn=./models/seqae.py model.name=SeqAELSTSQ model.args.dim_m=512 model.args.dim_a=8 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/vol21/masomatics/result/ulas/20230502_bigdim_0/T4_dim_a8" | kubectl create -f -


python new_minai_workflow.py --option "--config_path=./configs/shiftim/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=50000 report_freq=10 model_snapshot_freq=50000 manager_snapshot_freq=50000 num_workers=8 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/shift_img.py train_data.name=Shift_cifar train_data.args.root=/mnt/vol21/hayasick/data/ train_data.args.train=True train_data.args.T=4 train_data.args.max_T=9 train_data.args.max_vshift=[-5,5] train_data.args.max_hshift=[-5,5] train_data.args.deform=True model.fn=./models/seqae.py model.name=SeqAELSTSQ model.args.dim_m=512 model.args.dim_a=4 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=40000 training_loop.args.reconst_iter=0 log_dir=/mnt/vol21/masomatics/result/ulas/20230502_bigdim_0/T4_dim_a4" | kubectl create -f -


