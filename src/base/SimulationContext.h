#pragma once

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/spin_mutex.h>

#include <random>
#include <string>
#include <utility>
#include <vector>

#include "neighborlist.hpp"
#include "types/particle.hpp"

struct SimulationParameters {
  std::vector<double> growth_rates;
  std::vector<double> division_lengths;
  std::vector<double> thickness_arr;

  // anisotropic friction
  double ANISOTROPIC_FRICTION;
  double CarryingCapacity, RateAdjust;

  std::string initial_type;
  int initial_cells, frameSize, tracers;
  int annulus_start_delay;
  int cpu_cores;
  int rng_seed;

  // system width
  double systemSize;
  double del_t;
  double diffusion_rot, diffusion_trans;
  double eff_diff_rot, etdiff;
  double f_0;
  size_t maxPtcls;
  std::string work_dir;
};

class SimulationContext : public SimulationParameters {
 public:
  SimulationContext(const SimulationParameters& params);

  Particle cellDivision(int i);
  inline bool isDividing(int i) const;
  void updatePtclGrid();
  bool updatePtcls();
  void compute_force(const NeighborList& nlist);

  const std::vector<Particle>& get_particles() const { return ptcl; }
  void setPtcl(std::vector<Particle>& particles) { ptcl = std::move(particles); }

 public:
  std::vector<Particle> ptcl;
  double max_overlap = 0.0;
  double max_disp = 0.0;
  bool rebuild_flag = false;
  tbb::enumerable_thread_specific<std::mt19937> tls_rng;

 private:
  std::vector<tbb::mutex> ptcl_mutexes;
};
