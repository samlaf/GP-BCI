#!/bin/bash
#SBATCH -o /network/tmp1/laferris/slurm-%j.out

## this script is called as
## sbatch chruns2d_script.sh <emg> <dt>
## where <emg> is optional, and defaults to 2

if [ -z "$1" ]
then
  EMG=2
else
  EMG=$1
fi
if [ -z "$2" ]
then
  DT=2
else
  DT=$2
fi
if [ -z "$3" ]
then
  UID=$SLURM_JOBID
else
  UID=$3-$SLURM_JOBID
fi

source /network/home/laferris/.bashrc
conda activate gp
python chruns_2d.py --uid $UID --emg $EMG --dt $DT