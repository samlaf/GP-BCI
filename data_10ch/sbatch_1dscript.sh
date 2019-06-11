uid=0
for ardkern in '--ardkern' ''; do
    for k in 2 3 4 5 6; do
	name="1d$ardkern--k=$k"
	options="$ardkern --k $k"
	sbatch --job-name $name chruns1d_script.sh $uid "$options"
    done
done
