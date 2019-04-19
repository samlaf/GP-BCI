# script can be called with a jobid to ask info about
# If not given, then take last one in sacct list
if [ ! $# -eq 0 ]; then
    jobid=$1
else
    jobid=$(sacct -u $USER --starttime=0414 | tail -n 1 | cut -c1-5)
fi

echo "Printing report for job" $jobid
cat slurm-$jobid.out
cat /network/tmp1/laferris/slurm-$jobid.out
