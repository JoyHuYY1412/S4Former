#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres gpu:2
#SBATCH -o /Logs/slurm-%A_%a.out
#SBATCH -t 8:00:00

#SBATCH -a 1-30%1

if [ -e ./work_is_done ]; then
    echo "work is done"
    scancel -t PENDING $SLURM_ARRAY_JOB_ID
    exit
fi


cmd="./tools/dist_train.sh \
configs/setr/setr_deit-base_pup_bs_8_512x512_80k_pascal_1over16_split_classic_sup.py 2 \
--seed 1999 \
--auto-resume"

# print start time and command to log
echo $(date)
echo $cmd

# start command
$cmd