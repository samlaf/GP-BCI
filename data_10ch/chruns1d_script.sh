#!/bin/bash
#SBATCH -o /network/tmp1/laferris/slurm-%j.out

## this script is called as
## sbatch chruns1d_script.sh <uid> <options string>

if [ -z "$1" ]
then
    uid=$SLURM_JOBID
else
    #UID=$3-$SLURM_JOBID
    uid=$1
fi

source /network/home/laferris/.bashrc
conda activate gp
# $2 should be a string of options
python chruns_1d.py --uid $uid --jobid $SLURM_JOBID $2
