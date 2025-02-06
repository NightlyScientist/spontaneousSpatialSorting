#ifndef MDMAIN
#define MDMAIN

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>

#include "dataManager.h"
#include "interactions.h"
#include "populations.h"
#include "timestep.h"

#define TRUE 1
#define FALSE 0

#define pi 3.141592654
#define SMALL 10e-16

void main_loop();

struct ptcl_var {
  double x, y, z;
  double ex, ey, ez;
  double force_x, force_y, force_z;
  double torque_x, torque_y, torque_z;
  double length;
  double growth_rate;
  double phi;
  int active;
  int ancestor;
  int splits;
  int ancestor_rnd_group;
  int ind_group;
  int ancestor_list[30];
  int color;
  int color2;
  int label;
  int idx_1, idx_2, box_index;
  bool can_grow;
};

struct Parameters {
  double growthRate;
  double thickness, thickness_sqrt, thickness_pow2, thickness_ulim;
  int cycles, save_step;
  int cores;
};

extern struct Parameters *prms;
extern struct ptcl_var *ptcl;

extern double growth_rates[2];
extern double division_lengths[2];

extern double thickness;
extern double thickness_sqrt;
extern double thickness_pow2;
extern double thickness_ulim;

extern int cycles, frameSize;
extern int lineBoxes, gridBoxes;
extern int n_active;

// a.s.c. -> when to start counting ancestors, else random labels
extern int ancestor_start_count;
extern int total_anscestors;
extern int ind_group_number;
extern int remove_token;
extern int initial_type;
extern int initial_cells;
extern int annulus_start_delay;

// system width
extern double systemSize;
extern double scale;

extern double del_t;
extern double diffusion_rot;
extern double diffusion_trans;
extern double eff_diff_rot;
extern double eff_diff_trans;
extern double area_particles;
extern double maxArea;

extern int maxPtcls;
extern int N_removed;
extern int *list;
extern int *head;
extern int *removed_list;

extern int tracers;
extern int recycleCells;
extern double forceConst;

extern std::string work_dir;

#endif /*MDMAIN*/
