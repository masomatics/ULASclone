# ULAS
Reproducing code for the paper: Unsupervised Learning of Algebraic Structure on Stationary Time Sequences

## The main implementation
Please see 
- the class `SeqAELSTSQ` in `./models/seqae.py` for the implementation of the proposed model.
- the function `tracenorm_of_normalized_laplacian` in `./utils/laplacian.py` for the implementation of the block diagonalization loss defined in Eq.(6) in our paper.

## Prerequisite
python3.7, CUDA11.2, cuDNN

Please install additional python libraries by:
```
pip install -r requirements.txt
```

## Download datasets 
Download compressed dataset files from the following link and decompress them into `/tmp/path/to/data_dir`

https://drive.google.com/drive/folders/1_uXjx06U48to9OSyGY1ezqipAbbuT0vq?usp=sharing

This is an example script to download and decompress the files:
```
# If gdown is not installed:
pip install gdown

export DATADIR_ROOT=/tmp/path/to/datadir/
gdown --folder https://drive.google.com/drive/folders/1_uXjx06U48to9OSyGY1ezqipAbbuT0vq?usp=sharing -C $DATADIR_ROOT
tar xzf  ${DATADIR_ROOT}/MNIST.tar.gz -C $DATADIR_ROOT
tar xzf  ${DATADIR_ROOT}/3dshapes.tar.gz -C $DATADIR_ROOT
tar xzf  ${DATADIR_ROOT}/smallNORB.tar.gz -C $DATADIR_ROOT
```

## Training with the proposed method
1. Select a config file for the dataset on which you want to train the model:
```
# On sequential MNIST
export CONFIG=configs/mnist/lstsq/lstsq.yml
# On sequential MNIST-bg
export CONFIG=configs/mnist/lstsq/lstsq.yml
# On 3DShapes
export CONFIG=configs/3dshapes/lstsq/lstsq.yml
# On SmallNORB
export CONFIG=configs/smallNORB/lstsq/lstsq.yml
# On accelerated sequential MNIST
export CONFIG=configs/mnist_accl/lstsq/holstsq.yml
```

2. Run:
```
export LOGDIR=/tmp/path/to/logdir
export DATADIR_ROOT=/tmp/path/to/datadir
python run.py --config_path=$CONFIG --log_dir=$LOGDIR --attr train_data.args.root=$DATADIR_ROOT
```

## Training all models we tested in our experiments
```
export LOGDIR=/tmp/path/to/logdir
export DATADIR_ROOT=/tmp/path/to/datadir
bash training_allmodels.sh $LOGDIR $DATADIR_ROOT
```

## Evaluations
You can evaluate {generated images/ equivariance errors/ prediction errors} of trained models with the following ipython notebooks:
- Generated images: `gen_images.ipynb`
- Equivariance errors: `equivariance_error.ipynb`
- Prediction errors: `extrp.ipynb`
