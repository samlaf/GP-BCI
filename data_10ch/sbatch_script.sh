uid=$1
# for EMG in 0 2 4; do
#     for DT in 0 10 20 40 60 80 100; do
# 	sbatch chruns2d_script.sh $EMG $DT $uid
# 	sbatch chruns2d_script.sh $EMG $DT $uid sa
#     done
# done

# false is just a keyword. we patternmatch for the -- only
for symkern in 'false'; do
    for multkern in '--multkern' 'false'; do
	for ardkern in '--ardkern' 'false'; do
	    for sa in '--sa' 'false'; do
		for constrain in '--constrain' 'false'; do
		    sbatch chruns2d_emgdt_script.sh $symkern $multkern $ardkern $sa $constrain
		done
	    done
	done
    done
done
