for mode in sinetwo
do
    python ./datasets/so3_data.py --embed=${mode}
done

for mode in sinetwo
do
    python ./datasets/so3_data.py --embed=${mode} --shared=1
done


#python ./datasets/so3_data.py --embed=sinetwo