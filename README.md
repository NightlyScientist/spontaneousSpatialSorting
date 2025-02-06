# This Code is written to simulate a growing bacterial colony.

Project Simulation Code can be executed, e.g., using 

> python src/scripts/main.py --cycles 2000 --frame_size 5000 --growth_rate 0.01 0.02 --division_lengths 1 2 --cores 2 --time_step 0.001 --recompile

This will compile the c++ code (in src/base) and execute with the provided options. Optionally, the --recompile flag will recompile the c++ code before executing. Running with mutiple cores is done by settings --cores number_of_cores. In this example, a population of two species with growth rates {0.01, 0.02} and division lengths {1, 2} is simulated over 2000 cycles with each cycle taking 0.001 time units; data is saved every 5000 cycles.

## Input Configuration

To see available options that can be set as well as an explanation for each parameter, use 

> python src/scripts/main.py --help

### Populations

We support simulating bacterial growth of two species of different growth rates and division lengths, but equal thickness. Further versions will relax this latter constraint. This is done using ArgParse to append a sequence of values to a list. Thus, to specify the parameters of the second populations, use 

> python src/scripts/main.py --growth_rates rate_1 [rate_2] --division_lengths length_1 [length_2]

where the values in brackets are optional, and default to running the simulation with a single species if not provided.

## Slurm Execution

The simulation code can be executed on the slurm cluster by passing the --slurm flag to the main.py script. This option uses the slurm_main.sh script to submit and queue the code.

## Analysis

Find a collection of analysis routines in _investigations/_ directory. In particular,
have a first look at investigations/colony.py_ to see snapshots of the colony growth. This file can also be used to examine the nematic field, local heterozygosity and radial order. 

In order to make videos, the *animateGrowth_smoothed.py* file should be used. To modify the videos between fit-to-frame to constant frame size, look at *tools/graphics/snapshots_smoothed.py*. 

### Parameter space

The file *parameter_space.py*  generates a summary table of some measurement data extracted from specified paths in a dataset. It is recommended to be run first when analyzing a dataset. 

### Spatial Sorting During Equal Division-Time Growth

In order to determine the spatial distribution of cells at the periphery as a function of time, run *spatial_dist_paper.py*. To tweak the periphery threshold $w_p$, change the *distance_threshold* variable. 
To analyze the data, look at *spatial_dist_paper.ipynb*

### Radial order

To reproduce the radial order parameter figures, look at *radial_order_paper.ipynb*. 

### Two-point correlation function

To compute the two-point correlation function, look at *correlation_10.py*. To plot the results, look at *correlation_10_paper.ipynb* especially under the log-lin heading. 

### Trajectory tracking

To track the trajectories of cells that end up at the periphery or in the bulk, look at *ancestors.py* for the plotting of paths, *radial_traj_slurm.py* for the computation of probabilities, and *radial_trajectory.ipynb* for the plotting of probabilities. 

### Heterozygosity

To compute the heterozygosity and reproduce the figures, look at *heterozygosity_paper.ipynb*. The final cells in that notebook produce the figures from the article. 

### Equal Growth Rate

To investigate the colony with the equal growth rate condition, look at *equal_gr_investigation.ipynb*. Both normalized and unnormalized conditions are shown. 




