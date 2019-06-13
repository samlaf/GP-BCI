uid=1d
for ARD in '--ARD' ''; do
    for k in 2 3 4 5 6 7 8 9 10; do
	name="1d$ARD--k=$k"
	options="$ARD --k $k"
	sbatch --job-name $name chruns1d_script.sh $uid "$options"
    done
done
