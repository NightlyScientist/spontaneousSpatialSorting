#!/bin/bash
#SBATCH --job-name=a.m.c.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=workspace/logs/slurm/submission_%j.log

echo
hostname; pwd; date
echo

parser() {
    # default values
    cores=${cores:-"4"}
    cycles=${cycles:-"0"}
    max_ptcls=${max_ptcls:-"0"}
  
    frame_size=${frame_size:-"0"}
    system_size=${system_size:-"0"}
    time_step=${time_step:-"0"}
    thickness=${thickness:-"0"}

    tracers=${tracers:-"0"}
    diffusion_constant=${diffusion_constant:-"0"}
    force_constant=${force_constant:-"0"}

    initial_type=${initial_type:-"0"}
    initial_cells=${initial_cells:-"0"}
    fill_fraction=${fill_fraction:-"0"}
    recycle_cells=${recycle_cells:-"0"}

    division_length=${division_length:-"0"}
    growth_rate=${growth_rate:-"0"}

    rng_seed=${rng_seed:-"0"}
    save_path=${save_path:-"0"}

    # Assign the values given by the user
    while [ $# -gt 0 ]; do
        if [[ $1 == *"--"* ]]; then
            param="${1/--/}"
            declare -g $param="$2"
        fi
        shift
    done
}

# get cli options
parser $@

echo "Running your script now"
echo

./src/base/build/main.exe $cores $cycles $max_ptcls $frame_size $system_size $time_step $thickness $tracers $diffusion_constant $force_constant $initial_type $fill_fraction $recycle_cells $division_length $growth_rate $initial_cells $rng_seed $save_path

echo
date
