#!/bin/bash

#SBATCH -p gpu22
#SBATCH --gres gpu:1
#SBATCH -o /Logs/slurm-%A_%a.out
#SBATCH -t 8:00:00

#SBATCH -a 1-30%1

if [ -e ./work_is_done ]; then
    echo "work is done"
    scancel -t PENDING $SLURM_ARRAY_JOB_ID
    exit
fi

cmd="python tools/get_flops.py ./configs/segformer/segformer_mit-b4_bs_8_512x512_80k_pascal_1over16_split_CPS_semi_beta_1_th_0.9_MT.py --shape 512"


# print start time and command to log
echo $(date)
echo $cmd

# start command
$cmd