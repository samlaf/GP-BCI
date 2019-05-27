#!/bin/bash
#SBATCH -o /network/tmp1/laferris/slurm-%j.out

## this script is called as
## sbatch chruns2d_script.sh <emg> <dt> <uid> <--dtprior>
## where <emg> is optional, and defaults to 2

if [ -z "$1" ]
then
  EMG=4
else
  EMG=$1
fi
if [ -z "$2" ]
then
  DT=60
else
  DT=$2
fi
if [ -z "$3" ]
then
    uid=$SLURM_JOBID
else
    #UID=$3-$SLURM_JOBID
    uid=$3
fi

source /network/home/laferris/.bashrc
conda activate gp
# $4 should be a string of options
python chruns_dt2d.py --syn 0 4 --dts 20 40 60 --uid $uid --jobid $SLURM_JOBID $4
