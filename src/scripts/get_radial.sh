#!/bin/bash
#SBATCH --job-name=amc_prm
#SBATCH --ntasks=1
#SBATCH --time=24:0:0
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=workspace/logs/slurm/submission_%j.log

echo
hostname; pwd; date
echo

echo "Running your visualization Python script now"
echo

python3 /home/mratman1/activeMatterCommunities/investigations/radial_traj_slurm.py

echo
date