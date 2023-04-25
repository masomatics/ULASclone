#for mode in orth orth_mlp orth_mlp_two
for mode in orth
do
    bash jobs/20220826_Mstar_${mode}/write_shell.sh
    bash jobs/20220826_Mstar_${mode}/Mstar_${mode}_pfkube.sh
done

#for mode in inv_reg inv_reg_cnn
for mode in inv_reg_cnn
do
    bash jobs/20220829_Mstar_${mode}/write_shell.sh
    bash jobs/20220829_Mstar_${mode}/Mstar_${mode}_pfkube.sh
done


