for mode in iResNet Linear MLP
do
    python ./datasets/so3_data.py --embed=${mode}
done