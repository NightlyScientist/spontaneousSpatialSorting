#!/bin/bash
#SBATCH --job-name=bdg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1000:00:00
#SBATCH --output=workspace/logs/slurm/submission_%j.log

echo
hostname; pwd; date
echo

parser() {
    # default values
    cores=${cores:-"4"}
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

    division_length=${division_length:-"0"}
    growth_rate=${growth_rate:-"0"}
    carrying_capacity=${carrying_capacity:-"1"}
    rate_adjust=${rate_adjust:-"1"}
    debug=${debug:-"0"}

    rng_seed=${rng_seed:-"0"}
    save_path=${save_path:-"0"}
    annulus_start_delay=${annulus_start_delay:-"200"}
    anisotropic_friction=${anisotropic_friction:-"0"}

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

args=(
  --maxPtcls $max_ptcls
  --dir $save_path 
  --seed $rng_seed
  --frameSize $frame_size
  --systemSize $system_size
  --del_t $time_step
  --ntracers $tracers
  --thickness $thickness
  --diffusion_trans $diffusion_constant
  --force $force_constant
  --annulus_start_delay $annulus_start_delay
  --initialCells $initial_cells
  --anisotropic_friction $anisotropic_friction
  --rate_adjust $rate_adjust
  --carrying_capacity $carrying_capacity
  --initial_type $initial_type
  --CPUs $cores
  --growth_rates $growth_rate
  --division_lengths $division_length
)

echo "${args[@]}" 

#./src/base/build/main.exe $cores $cycles $max_ptcls $frame_size $system_size $time_step $thickness $tracers $diffusion_constant $force_constant $initial_type $fill_fraction $recycle_cells $division_length $growth_rate $initial_cells $rng_seed $save_path $annulus_start_delay $anisotropic_friction $rate_adjust $carrying_capacity

if [ "$debug" -eq 1 ]; then
  gdb --args ./src/base/build/main.exe "${args[@]}"
else
  ./src/base/build/main.exe "${args[@]}"
fi

echo
date
