uid=latest2
for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for sa in '--sa' ''; do
	    for constrain in '--constrain' ''; do
		sbatch chruns2d_script.sh 4 60 $uid "$multkern $ardkern $sa $constrain"
		sbatch chruns2d_script.sh 0 10 $uid "$multkern $ardkern $sa $constrain"
	    done
	done
    done
done

for multkern in '--multkern' ''; do
    for ardkern in '--ardkern' ''; do
	for sa in '--sa' ''; do
	    for constrain in '--constrain' ''; do
		sbatch chruns2d_script.sh 0 0 $uid "--symkern $multkern $ardkern $sa $constrain"
		sbatch chruns2d_script.sh 4 0 $uid "--symkern $multkern $ardkern $sa $constrain"
	    done
	done
    done
done

