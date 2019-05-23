uid=dt2d
for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for sa in '--sa' ''; do
	    for constrain in '--constrain' ''; do
		name="dt2d$multkern$ardkern$sa$constrain"
		options="$multkern $ardkern $sa $constrain"
		sbatch --job-name $name chrunsdt2d_script.sh 4 60 $uid $options
	    done
	done
    done
done
