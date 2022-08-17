for mode in iResNet Linear MLP
do
    python ./datasets/so3_data.py --embed=${mode} --datamode=so2 --num_blocks=3 --recreate=1
done

for mode in iResNet Linear MLP
do
    python ./datasets/so3_data.py --embed=${mode} --datamode=so2 --num_blocks=3 --shared=1
done