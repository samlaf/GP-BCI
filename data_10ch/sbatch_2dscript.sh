uid=2d
for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for k in 2 4 6; do
	    name="2d$multkern$ardkern--k=$k"
	    options="$multkern $ardkern --k $k"
	    sbatch --job-name $name chruns2d_script.sh 4 60 $uid "$options"
	    sbatch --job-name $name chruns2d_script.sh 0 10 $uid "$options"
	done
    done
done

for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for k in 2 4 6; do
	    name="2d--symkern$multkern$ardkern$--k=$k"
	    options="--symkern $multkern $ardkern --k $k"
	    sbatch --job-name $name chruns2d_script.sh 0 0 $uid "$options"
	    sbatch --job-name $name chruns2d_script.sh 4 0 $uid "$options"
	done
    done
done

