for mode in iResNet Linear MLP
#for mode in Linear
do
    python ./datasets/so3_data.py --embed=${mode}
done

for mode in iResNet Linear MLP
#for mode in Linear
do
    python ./datasets/so3_data.py --embed=${mode} --shared=1
done