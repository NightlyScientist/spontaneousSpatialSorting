#include <math.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/mutex.h>
#include <oneapi/tbb/parallel_for.h>
#include <stdio.h>

#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "SimulationContext.h"
#include "interactions.h"
#include "types/coordinates.hpp"
#include "types/particle.hpp"

// Thread-safe exponential distribution
inline double exp_dist(double diff_t, std::mt19937& rng) {
  std::exponential_distribution<double> exp_dist(1.0 / diff_t);
  return exp_dist(rng);
}

// Thread-safe Gaussian distribution
inline double gauss(double sigma, double mu, std::mt19937& rng) {
  std::normal_distribution<double> normal(mu, sigma);
  return normal(rng);
}

// Anisotropic friction
inline Vec2 applyFriction(const Particle& ptcl, double f_0, double ANISOTROPIC_FRICTION) {
  const double fx = ptcl.forces[0], fy = ptcl.forces[1];
  double ex = ptcl.ori[0], ey = ptcl.ori[1];
  double vx, vy;

  if (ANISOTROPIC_FRICTION != 1.0) {
    const double kper = ANISOTROPIC_FRICTION;
    const double kpar = 1.0 / ANISOTROPIC_FRICTION;

    // Precompute reused sub-expressions
    double ex2 = ex * ex;
    double ey2 = ey * ey;
    double exey = ex * ey;
    double fxkper = fx * kper;
    double fxkpar = fx * kpar;
    double fykper = fy * kper;
    double fykpar = fy * kpar;

    vx = f_0 * (fxkper * ex2 - fykpar * exey + fykper * exey + fxkpar * ey2);
    vy = f_0 * (fykpar * ex2 - fxkpar * exey + fxkper * exey + fykper * ey2);

  } else {
    vx = fx * f_0;
    vy = fy * f_0;
  }
  return {vx, vy};
}

inline bool SimulationContext::isDividing(int i) const {
  const int label = static_cast<int>(ptcl[i].type);
  return ptcl[i].can_grow && (label < 2) && (ptcl[i].length >= division_lengths[label]);
}

Particle SimulationContext::cellDivision(int i) {
  Particle daughter = ptcl[i];
  auto& r = ptcl[i];
  const auto p_type = static_cast<int>(r.type);

  // update resulting cell positions
  r.pos += 0.25 * r.length * r.ori;
  daughter.pos -= 0.25 * r.length * r.ori;

  r.length = daughter.length = r.length / 2.0;

  // update growth properties
  daughter.growth_rate = growth_rates[p_type] * std::uniform_real_distribution<double>(0.5, 1.5)(tls_rng.local());

  // update ancestor history
  r.ancestor_list.push_back(r.ancestor);
  daughter.ancestor_list.push_back(r.ancestor);

  // update ancestor labels
  r.ancestor = i;
  daughter.ancestor = i;
  daughter.ancestor_rnd_group = r.ancestor_rnd_group;

  // Update positions and lengths
  r.splits += 1;
  daughter.splits = 0;
  return daughter;
}

// Particle update with parallelism and thread-safe RNG
bool SimulationContext::updatePtcls() {
  tbb::concurrent_vector<Particle> new_rods;

  tbb::enumerable_thread_specific<int> tls_stale_count([]() { return 0; });
  tbb::enumerable_thread_specific<int> tls_max_disp([]() { return 0; });

  tbb::parallel_for(tbb::blocked_range<int>(0, ptcl.size()), [&](const tbb::blocked_range<int>& r) {
    auto& rng = tls_rng.local();
    auto& local_stale_count = tls_stale_count.local();
    auto& local_max_disp = tls_max_disp.local();

    for (int i = r.begin(); i != r.end(); ++i) {
      auto& p = ptcl[i];

      const int label = static_cast<int>(ptcl[i].type);
      double rescaled_f0 = f_0 * (0.1 / thickness_arr[label]);
      double o_length = 1.0 / p.length;
      double o_length3 = o_length * o_length * o_length;

      // applies friction based on ANISOTROPIC_FRICTION (see
      // 0.1098/rsif.2017.0073)
      auto vel = applyFriction(p, rescaled_f0, ANISOTROPIC_FRICTION);

      if (etdiff != 0.0) {
        const std::array<double, 2> diffusion = {etdiff * gauss(1.0, 0.0, rng), etdiff * gauss(1.0, 0.0, rng)};
        p.pos += o_length * del_t * vel + diffusion;
      } else {
        p.pos += o_length * del_t * vel;
      }

      // Update orientation
      if (eff_diff_rot != 0.0) {
        double rnd_phi = eff_diff_rot * gauss(1.0, 0.0, rng);
        p.phi += rescaled_f0 * 12.0 * p.torque * o_length3 * del_t + rnd_phi;
      } else {
        p.phi += rescaled_f0 * 12.0 * p.torque * o_length3 * del_t;
      }

      p.ori = {std::cos(p.phi), std::sin(p.phi)};

      // Increase particle length if possible
      if (!p.stale) {
        p.length += p.growth_factor * p.growth_rate * del_t;
      }

      // Cell division (protected by mutex)
      if (isDividing(i)) {
        new_rods.push_back(cellDivision(i));
      }

      // pbc(p.pos, systemSize);
      double disp = norm(p.pos - p.pos_last_update);
      if (disp > local_max_disp) local_max_disp = disp;

      p.pos_last_update = p.pos;
    }
  });

  max_disp = 0;
  for (const auto& local_max : tls_max_disp) {
    if(local_max > max_disp) max_disp = local_max;
  }

  int stale_count = 0;
  for (const auto& local_count : tls_stale_count) {
    stale_count += local_count;
  }

  if (stale_count >= static_cast<int>(ptcl.size())) {
    printf("  ! all particles can longer grow, ending simulation.\n");
    return false;
  }

  if (new_rods.size() > 0) {
    for (auto& p : new_rods){
      p.pos_last_update = p.pos;
    }
    ptcl.insert(ptcl.end(), new_rods.begin(), new_rods.end());
    rebuild_flag = true;
  }

  if (ptcl.size() >= maxPtcls) {
    printf("  ! particle amount exceeded, ending simulation.\n");
    return false;
  }

  return true;
}
