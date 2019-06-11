# uid=dt2d4
# for multkern in '--multkern' ''; do
#     for ardkern in '--ardkern' ''; do
# 	for sa in '--sa' ''; do
# 	    for npq in 0 3; do
# 		name="dt2d$multkern$ardkern$sa$constrain--npq$npq"
# 		options="$multkern $ardkern $sa --constrain --n_prior_queries $npq"
# 		sbatch --job-name $name chrunsdt2d_script.sh 4 60 $uid "$options"
# 	    done
# 	done
#     done
# done

uid=dt2dkmult
for ardkern in '--ardkern' ''; do
    for k in 2 3 4 5 6; do
	name="dt2dk--multkern$ardkern--constrain--k$k"
	options="--multkern $ardkern --constrain --k $k"
	sbatch --job-name $name chrunsdt2d_script.sh 4 60 $uid "$options"
    done
done
