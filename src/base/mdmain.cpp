#include "mdmain.h"

#include <math.h>
#include <omp.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <time.h>

#include <iostream>
#include <string>

#include "dataManager.h"
#include "interactions.h"
#include "populations.h"
#include "timestep.h"

// cell constants
double thickness;
double thickness_sqrt;
double thickness_pow2;
double thickness_ulim;

// simulation area constants
int cycles;
int frameSize, n_active;
int lineBoxes, gridBoxes;
int ancestor_start_count = 25;
int Start_parts;
int ind_group_number = 0;
int total_anscestors = 2;
int N_removed = 0;
int remove_token = 1;
int annulus_start_delay;

double del_t;
double systemSize, scale;
double diffusion_rot, diffusion_trans;
double eff_diff_rot, eff_diff_trans;
double area_particles = 0;
double maxArea;

double growth_rates[2];
double division_lengths[2];

int initial_cells, initial_type;

// structs
struct ptcl_var *ptcl;
struct Parameters *prms;

double forceConst;
int maxPtcls;
int tracers;
int recycleCells;
std::string work_dir;

// arrays
int *list;
int *head;
int *removed_list;

void main_loop() {
  periodic_boundary_condition();
  generate_linked_cell_list();

  int next_milestone = (int)(n_active * 1.05), doubling_milestone = n_active;
  int save_counter = 1, stop_simulation = 0;
  double time = 0;
  int save_interval = 1;
  bool added_annulus = (initial_type == 3) ? false : true;

  while (stop_simulation == 0) {
    stop_simulation = timeStep();
    time += del_t;

    if (added_annulus == false && n_active > annulus_start_delay) {
      annulus(n_active, false, 1, true);
      added_annulus = true;
    }

    periodic_boundary_condition();
    generate_linked_cell_list();

    if (stop_simulation == 1) {
      save_data(cycles, time);
      // save_data(save_counter, time);
      break;
    }

    // if (n_active >= next_milestone) {
    if (save_interval == frameSize) {
      save_data(cycles, time);
      // save_data(save_counter, time);
      // next_milestone = (int)(n_active * 1.05);
      // save_counter++;
      save_interval = 0;
    }

    // if (n_active >= 2000 && n_active >= doubling_milestone) {
    //   doubling_milestone = 2 * n_active;
    //   del_t = del_t / 2;
    // }
    save_interval++;
    cycles++;
  }
}

int main(int argc, char *argv[]) {
  // read parameters from input.dat 'config file'
  read_parameters(argc, argv);

  int cores = (int)atoi(argv[1]);
  omp_set_num_threads(cores);
  printf("using %d cores\n", cores);

  // set seed val to pseudo rng
  int rng_seed = atoi(argv[17]);
  printf("using %d seed\n", rng_seed);
  srand48(rng_seed);

  // assert that the number of initial_cell is less than the maxPtcls
  if (initial_cells > maxPtcls) {
    printf("\t!! initial_cells is greater than maxPtcls\n");
    exit(1);
  }

  // allocate memory for arrays
  ptcl = (struct ptcl_var *)calloc(maxPtcls, sizeof(struct ptcl_var));
  list = (int *)calloc(maxPtcls, sizeof(int));
  head = (int *)calloc(gridBoxes, sizeof(int));
  removed_list = (int *)calloc(maxPtcls, sizeof(int));

  // initial condition for the cells
  starting_cells();

  // save ptcl data at initial time
  save_data(0, 0.0);

  // // start time
  time_t l_current_t = time(nullptr);
  printf("starting time: %s\n", asctime(gmtime(&l_current_t)));

  main_loop();

  // end time
  l_current_t = time(nullptr);
  printf("end time: %s\n", asctime(gmtime(&l_current_t)));

  return 0;
}
