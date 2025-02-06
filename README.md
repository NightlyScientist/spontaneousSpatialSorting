[![DOI](https://zenodo.org/badge/923148635.svg)](https://doi.org/10.5281/zenodo.14827698)


## Agent-based simulations of growing bacterial colonies with coexisting shape phenotypes.
This repository is asssociated with the preprint, Ratman et al., *"Spontaneous spatial sorting by cell shape in growing colonies of rod-like bacteria"* (2025), which can be found on [arXiv](https://doi.org/10.48550/arXiv.2501.11177) and [bioRxiv](https://doi.org/10.1101/2025.01.22.634274).

Project simulation code can be executed using, for example,

>  python src/scripts/main.py --frame_size 5000 --cores 4 --time_step 0.001 --diffusion_constant 0 --max_aspect_ratio 4 4 --initial_type uniform --max_ptcls 1000 --equal_division_time --rng_seed 1 --number_experiments 1 --save_path workspace/simulations/ --recompile --growth_rate 0.01
 

This will compile the C++ code (in src/base) and execute with the provided options. Optionally, the `--recompile` flag will recompile the C++ code before executing. Running with mutiple cores is done by settings `--cores` number_of_cores. In this example code, a population of two species with growth rate 0.01 and division lengths {4, 4} is simulated until 1000 cells are generated, with each cycle taking 0.001 time units; data is saved every 5000 cycles. Note that if `--number_experiments`, which determines the number of simulations of different seeds to be executed, is larger than 1, simulations will be sent to a slurm scheduler. The `--equal_division_time` flag ensures that both populations have the same division time by modifying the growth rate of the second population, otherwise both populations are given the same growth rate set by `--growth_rate`.

## Input configuration
To see available options that can be set as well as an explanation for each parameter, use

 
> python src/scripts/main.py --help
 

## Populations
We support simulating bacterial growth of two species of different growth rates and division lengths, but equal thickness. Further versions will relax this latter constraint. This is done using ArgParse to append a sequence of values to a list. Thus, to specify the parameters of the second populations, use

 

> python src/scripts/main.py --max_aspect_ratio aspect_ratio_1 [aspect_ratio_2] --growth_rate rate
 

where the values in brackets are optional, and default to running the simulation with a single species if not provided.

## Slurm execution
The simulation code can be executed on a Slurm cluster by passing the `--slurm` flag to the main.py script. This option uses the slurm_main.sh script to submit and queue the code.

## Analysis
A collection of analysis routines are in the *investigations/* directory. In particular, investigations/colony.py generates snapshots of the colony growth. This file can also be used to examine the nematic director field, local heterozygosity, and radial order.

Videos can be generated using animateGrowth_smoothed.py. Smoothed animations with a dynamic or constant size of viewing window can be generated using [snapshots_smoothed](investigations/tools/graphics/snapshots_smoothed.py).

### Parameter space
The file parameter_space.py generates a summary table of some measurement data extracted from specified paths in a dataset. It is recommended to be run first when analyzing a dataset.

### Spatial sorting during equal division-time growth
In order to determine the spatial distribution of cells at the periphery as a function of time, run spatial_dist_paper.py. To tweak the periphery threshold wp, change the distance_threshold variable. To analyze the data, look at [spatial_dist_paper.ipynb](investigations/spatial_dist_paper.ipynb).

### Radial order
The notebook radial_order_paper.ipynb generates radial order parameter plots.

### Two-point correlation function
To compute the two-point cell type correlation function, see [correlation_10.py](investigations/correlation_10.py). To plot the results, see [correlation_10_paper.ipynb](investigations/correlation_10_paper.ipynb) especially under the "log-lin" heading.

### Trajectory tracking
To track the trajectories of cells, see ancestors.py for the plotting of paths, [radial_traj_slurm.py](investigations/radial_traj_slurm.py) for the computation of probabilities, and radial_trajectory.ipynb for the plotting of probabilities.

### Heterozygosity
To compute the heterozygosity and reproduce the figures, see [heterozygosity_paper.ipynb](investigations/heterozygosity_paper.ipynb). The final cells in that notebook produce the figures from the article.

### Equal growth rate
To investigate the colony with the equal growth rate condition, see [equal_gr_investigation.ipynb](investigations/equal_gr_investigation.ipynb). Both normalized and unnormalized conditions are shown.


## Package Dependencies

This code requires python (v3.11+) to be installed. Our Python environment is provided in [requirements.txt](requirements.txt).

