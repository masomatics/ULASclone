python ./run.py --config_path=./configs/shiftim/lstsq/lstsq.yml --attr batchsize=32 seed=1 max_iteration=60000 report_freq=10 model_snapshot_freq=60000 manager_snapshot_freq=60000 num_workers=2 T_cond=2 T_pred=1 lr=0.0003 reg.reg_bd=None reg.reg_orth=None train_data.fn=./datasets/shift_img.py train_data.name=Shift_cifar train_data.args.root=/mnt/vol21/hayasick/data/ train_data.args.train=True train_data.args.T=6 train_data.args.max_T=9 train_data.args.max_vshift=[-10,10] train_data.args.max_hshift=[-10,10] train_data.args.deform=True model.fn=./models/seqae.py model.name=SeqAELSTSQ model.args.dim_m=256 model.args.dim_a=4 model.args.ch_x=3 model.args.k=2.0 model.args.predictive=True model.args.detachM=0 training_loop.fn=./training_loops.py training_loop.name=loop_seqmodel training_loop.args.lr_decay_iter=50000 training_loop.args.reconst_iter=40000 log_dir=/mnt/vol21/masomatics/result/ulas/20230503_fisheyefour_0/reconst_iter40000
