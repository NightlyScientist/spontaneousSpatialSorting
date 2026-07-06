#include <math.h>
#include <omp.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/global_control.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>
#include <cstdio>
#include <iostream>
#include <random>
#include <utility>

#include "SimulationContext.h"
#include "dataManager.h"
#include "neighborlist.hpp"
#include "nutrientField.hpp"
#include "populations.hpp"

void main_loop(SimulationContext& ctx, NeighborList& nlist, BoxAssignment& cells, NutrientField& nf) {
  const auto annulus_start_delay = ctx.annulus_start_delay;
  const auto frameSize = ctx.frameSize;
  auto timestep = ctx.del_t;

  int low_overlap_counter = 0;             // Counts consecutive low-overlap steps
  const int hysteresis_steps = 5;          // Number of consecutive steps below lower threshold before increasing dt
  const double min_dt = 0.01 * ctx.del_t;  // Smallest allowed time step
  const double max_dt = 100 * ctx.del_t;  // Largest allowed time step
  const double base_overlap_unit = std::max(ctx.thickness_arr[0], ctx.thickness_arr[1]) / 2;
  double overlap_upper = base_overlap_unit;        // Upper overlap threshold
  double overlap_lower = 0.1 * base_overlap_unit;  // Lower overlap threshold
  int division_counter = 2 * ctx.ptcl.size();

  double time = 0;
  bool added_annulus = (ctx.initial_type == "passive_ring") ? false : true;
  bool continue_simulation = true;
  int iteration = 0, save_interval = 1;
  auto& max_overlap = ctx.max_overlap;

  std::cout << "nlist cutoffs: " << nlist.cutoff_with_skin[0][0] << " " << nlist.cutoff_with_skin[0][1] << " " << nlist.cutoff_with_skin[1][1] << std::endl;
  std::cout << "skin: " << nlist.skin << std::endl;
  std::cout << "cellSize: " << cells.cellSize << std::endl;

  // force a rebuild of the neighbor list
  ctx.rebuild_flag = true;

  while (continue_simulation) {

    if (ctx.max_disp > nlist.skin || ctx.rebuild_flag) {
      if (ctx.rebuild_flag){
        // reset list for neighborlist and boxassignment
        nlist.resizeLists(ctx.ptcl.size());
        cells.resizeLists(ctx.ptcl.size());
        ctx.rebuild_flag = false;
      }

      cells.updateCellBoxes(ctx.ptcl);
      // nlist.build_neighbor_list_brute(ctx.ptcl);
      nlist.build_neighbor_list(cells, ctx.ptcl);
      ctx.rebuild_flag = false;
    }

    ctx.compute_force(nlist);
    continue_simulation = ctx.updatePtcls();

    nf.updateField(ctx.ptcl);
    continue_simulation = nf.updateParticles(ctx.ptcl) && continue_simulation;

    time += timestep;

    if (max_overlap > overlap_upper) {
      // Immediate reduction for safety
      timestep = std::max(timestep * 0.5, min_dt);
      low_overlap_counter = 0;
    } else if (max_overlap < overlap_lower) {
      low_overlap_counter++;

      if (low_overlap_counter >= hysteresis_steps) {
        timestep = std::min(timestep * 1.1, max_dt);
        low_overlap_counter = 0;
      }
    } else {
      low_overlap_counter = 0;
    }

    // if (added_annulus == false && ctx.ptcl.size() > annulus_start_delay) {
    //   annulus(ctx.ptcl.size(), ctx, false, true, 1);
    //   added_annulus = true;
    // }

    if (!continue_simulation) {
      save_data_csv(iteration, time, ctx);
      break;
    }

    if (save_interval == frameSize) {
      save_data_csv(iteration, time, ctx);
      save_interval = 0;
    }

    if (static_cast<int>(ctx.ptcl.size()) > division_counter) {
      division_counter *= 2;
      overlap_upper *= 1.05;
      overlap_lower *= 1.05;
      std::cout << "overlaps: " << overlap_lower << " < " << max_overlap << " < " << overlap_upper << std::endl;
    }

    save_interval++;
    iteration++;
  }
}

void setSeed(int seed) { srand48(seed); }

void setCPUCores(int cpu_cores) {
  omp_set_num_threads(cpu_cores);
  std::cout << "CPU Cores -> " << omp_get_max_threads() << std::endl;
}

std::pair<BoxAssignment, NeighborList> setupAuxList(SimulationContext& ctx){
  std::vector<double> l0 = ctx.division_lengths;  // rod lengths
  std::vector<double> d0 = ctx.thickness_arr; // rod thicknesses or related
  const double  skin = *std::min(d0.begin(), d0.end());

  // Calculate combined cutoffs same way as below:
  NeighborList neighborList(d0, l0, skin);
  neighborList.setSepMatrices(d0);

  double min_cutoff = neighborList.cutoff_with_skin[0][0];
  for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
          min_cutoff = std::min(min_cutoff, neighborList.cutoff_with_skin[i][j]);

  BoxAssignment box(min_cutoff, ctx.systemSize, ctx.ptcl.size());

  neighborList.resizeLists(ctx.ptcl.size());
  neighborList.build_neighbor_list(box, ctx.ptcl);
  return std::make_pair(box, neighborList);
}

int main(int argc, char* argv[]) {
  // read parameters from input.dat 'config file'
  SimulationParameters params;
  read_parameters(argc, argv, params);

  // set rng seed and cpu cores
  setCPUCores(params.cpu_cores);
  setSeed(params.rng_seed);
  tbb::global_control tbb_thread_limit(tbb::global_control::max_allowed_parallelism, params.cpu_cores);

  // assert that the number of initial_cell is less than the maxPtcls
  if (params.initial_cells > static_cast<int>(params.maxPtcls)) {
    printf("\t!! initial_cells is greater than maxPtcls\n");
    exit(1);
  }

  // create simulation context
  SimulationContext ctx(params);

  // Initialize thread-local RNGs with unique seeds
  const int base_seed = params.rng_seed;

  // Use an atomic counter for unique seeds per thread
  static std::atomic<unsigned int> thread_counter{0};
  ctx.tls_rng = tbb::enumerable_thread_specific<std::mt19937>([base_seed]() {
    unsigned int thread_id = thread_counter++;
    return std::mt19937(base_seed + thread_id);
  });

  // initial condition for the cells
  ctx.ptcl = starting_cells(params, base_seed);

  // now intiialize box and neighbor list
  auto [box, nlist] = setupAuxList(ctx);

  // nutrient field
  // NutrientField nf(box.nBoxes, params.systemSize);
  NutrientField nf(box.nBoxes, params.CarryingCapacity, params.RateAdjust, params.del_t);

  std::cout << " number of particles is " << ctx.ptcl.size() << std::endl;

  // save ptcl data at initial time
  save_data_csv(0, 0.0, ctx);

  // start time
  auto start = std::chrono::high_resolution_clock::now();

  // main event loop
  std::cout << "TBB max threads -> " << tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism)
            << std::endl;

  main_loop(ctx, nlist, box, nf);

  // end time
  auto end = std::chrono::high_resolution_clock::now();

  // print the simulation time in hours and minutes
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  int hours = duration / 3600;
  int minutes = (duration % 3600) / 60;
  printf("Total simulation time: %d hours %d minutes\n", hours, minutes);

  return 0;
}
