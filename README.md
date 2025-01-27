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
have a first look at investigations/colony.py_ to see snapshots of the colony growth. 

### Spatial Sorting During Equal Division-Time Growth
In 