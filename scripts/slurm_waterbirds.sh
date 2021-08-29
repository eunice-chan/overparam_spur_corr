#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=30 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ace # if you need specific nodes
#SBATCH -t 15-00:00 # time requested (D-HH:MM)

#SBATCH -o slurm.%j.%N.supcon_waterbirds_default.out # STDOUT
#SBATCH -e slurm.%j.%N.supcon_waterbirds_default.err # STDERR

#SBATCH -D /data/ekchan/repos/overparam_spur_corr/logs
cd /data/ekchan/repos/overparam_spur_corr

echo "DIRECTORY:"
pwd
echo "MACHINE:"
hostname
echo "START TIME:"
date

source ~/.bashrc
conda activate supcon

export PYTHONUNBUFFERED=1

python main_supcon.py --dataset waterbirds --num_workers 30 --size 224

wait
echo "END TIME:"
date