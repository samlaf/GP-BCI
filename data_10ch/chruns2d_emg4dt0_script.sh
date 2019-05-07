#!/bin/bash
#SBATCH -o /network/tmp1/laferris/slurm-%j.out

## this script is called as
## sbatch chruns2d_script.sh <emg> <dt> <uid> <--dtprior>
## where <emg> is optional, and defaults to 2

if [ $1 == "--symkern" ]
then
  symkern=$1
else
  symkern=''
fi
if [ $2 == '--multkern' ]
then
  multkern=$2
else
  multkern=''
fi
if [ $3 == '--ardkern' ]
then
    ardkern=$3
else
    ardkern=''
fi
if [ $4 == '--sa' ]
then
    sa="--sa"
else
    sa=""
fi

source /network/home/laferris/.bashrc
conda activate gp
python chruns_2d.py --emg 4 --dt 0 --uid 9306 $symkern $multkern $ardkern $sa
