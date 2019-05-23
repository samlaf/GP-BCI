uid=latest3
for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for sa in '--sa' ''; do
	    for constrain in '--constrain' ''; do
		name="dt$multkern$ardkern$sa$constrain"
		options="$multkern $ardkern $sa $constrain"
		sbatch --job-name $name chruns2d_script.sh 4 60 $uid $options
		sbatch --job-name $name chruns2d_script.sh 0 10 $uid $options
	    done
	done
    done
done

for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for sa in '--sa' ''; do
	    for constrain in '--constrain' ''; do
		name="dt--symkern$multkern$ardkern$sa$constrain"
		options="--symkern $multkern $ardkern $sa $constrain"
		sbatch --job-name $name chruns2d_script.sh 0 0 $uid $options
		sbatch --job-name $name chruns2d_script.sh 4 0 $uid $options
	    done
	done
    done
done

