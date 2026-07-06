// #pragma once
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

#include "SimulationContext.h"
#include "neighborlist.hpp"
#include "types/coordinates.hpp"
#include "types/particle.hpp"

struct ForceResult {
  double fx, fy, tz;
  double nbor_fx, nbor_fy, nbor_tz;
  int nbor_idx;
  bool has_nbor;
};

struct StericForceResult {
  double fx, fy, tz;
  double nbor_fx, nbor_fy, nbor_tz;
  int nbor_idx;
  bool has_nbor;
};

struct LocalForce {
  double fx, fy, tz;
  LocalForce() : fx(0), fy(0), tz(0) {}
};

StericForceResult steric_force(double h12, double hx, double hy, double lambda1, double lambda2, double ex1, double ey1,
                               double ex2, double ey2, int npidx, int interaction_type, double thickness) {
  StericForceResult res = {0, 0, 0, 0, 0, 0, npidx, false};

  double cx1 = -lambda1 * ex1 + hx * 0.5;
  double cy1 = -lambda1 * ey1 + hy * 0.5;
  double cx2 = -lambda2 * ex2 - hx * 0.5;
  double cy2 = -lambda2 * ey2 - hy * 0.5;

  double h = std::sqrt(h12);
  double hx_norm = hx / h;
  double hy_norm = hy / h;
  double h_gap = thickness - h;
  double h_gap3 = std::pow(h_gap, 3);
  double fi_str = std::sqrt(h_gap3);

  res.fx = fi_str * hx_norm;
  res.fy = fi_str * hy_norm;
  res.tz = fi_str * (cx1 * hy_norm - cy1 * hx_norm);

  if (interaction_type == 0) {
    res.nbor_fx = -fi_str * hx_norm;
    res.nbor_fy = -fi_str * hy_norm;
    res.nbor_tz = fi_str * (cx2 * hy_norm - cy2 * hx_norm);
    res.has_nbor = true;
  }
  return res;
}

void SimulationContext::compute_force(const NeighborList& nlist) {
  const int n_active = static_cast<int>(ptcl.size());

  // Zero out forces in parallel
  tbb::parallel_for(0, n_active, [&](int ptcl_idx) {
    ptcl[ptcl_idx].forces = {0.0, 0.0};
    ptcl[ptcl_idx].torque = 0.0;
  });

  // Thread-local overlap accumulators for each particle
  tbb::enumerable_thread_specific<double> tls_max_overlap([]() { return 0.0; });
  tbb::enumerable_thread_specific<double> tls_max_disp([]() { return 0.0; });

  // Thread-local force accumulators for each particle
  tbb::enumerable_thread_specific<std::vector<LocalForce>> tls_forces(
      [&]() { return std::vector<LocalForce>(n_active); });

  // Thread-local force accumulators for neighbor particles
  tbb::enumerable_thread_specific<std::vector<LocalForce>> tls_nbor_forces(
      [&]() { return std::vector<LocalForce>(n_active); });

  tbb::parallel_for(0, n_active, [&](int ptcl_idx) {
    auto& local_forces = tls_forces.local();
    auto& nbor_forces = tls_nbor_forces.local();
    auto& local_max_overlap = tls_max_overlap.local();

    auto& p1 = ptcl[ptcl_idx];
    double ex1 = p1.ori[0];
    double ey1 = p1.ori[1];

    local_max_overlap = 0;

    for (int n = 0; n < nlist.neighbor_count[ptcl_idx]; ++n) {
      int npidx = nlist.neighborlist[ptcl_idx][n];
      // if (npidx <= ptcl_idx) continue;

      auto& p2 = ptcl[npidx];
      auto dxdy = displacement(p1.pos, p2.pos);
      double dx = dxdy[0], dy = dxdy[1];

      const double ex2 = p2.ori[0];
      const double ey2 = p2.ori[1];
      const auto ptype_1 = static_cast<int>(p1.type);
      const auto ptype_2 = static_cast<int>(p2.type);
      const double combined_thickness_ulim = nlist.m_sep_ulim[ptype_1][ptype_2];
      const double combined_thickness = nlist.m_sep[ptype_1][ptype_2];
      const double combined_thickness_pow2 = nlist.m_sep_sqr[ptype_1][ptype_2];

      for (int orientation_sign = 0; orientation_sign < 2; orientation_sign++) {
        double lambda1 = (p1.length - p1.thickness) * 0.5;
        if (orientation_sign == 1) lambda1 = -lambda1;

        double lambda2 = (dx - lambda1 * ex1) * ex2 + (dy - lambda1 * ey1) * ey2;
        double hx = dx - lambda2 * ex2 - lambda1 * ex1;
        double hy = dy - lambda2 * ey2 - lambda1 * ey1;
        double h12 = hx * hx + hy * hy;
        double lambda2_prime = (p2.length - p2.thickness) * 0.5;

        if (h12 < combined_thickness_pow2 && h12 > combined_thickness_ulim &&
            lambda2 * lambda2 < lambda2_prime * lambda2_prime) {
          double overlap = combined_thickness - sqrt(h12);
          // if (overlap > local_max_overlap) local_max_overlap = overlap;
          local_max_overlap += overlap;

          auto result = steric_force(h12, hx, hy, lambda1, lambda2, ex1, ey1, ex2, ey2, npidx, 0, combined_thickness);
          local_forces[ptcl_idx].fx += result.fx;
          local_forces[ptcl_idx].fy += result.fy;
          local_forces[ptcl_idx].tz += result.tz;
          if (result.has_nbor && result.nbor_idx >= 0 && result.nbor_idx < n_active) {
            nbor_forces[result.nbor_idx].fx += result.nbor_fx;
            nbor_forces[result.nbor_idx].fy += result.nbor_fy;
            nbor_forces[result.nbor_idx].tz += result.nbor_tz;
          }
        } else {
          lambda2 = lambda2_prime;
          hx = dx - lambda2 * ex2 - lambda1 * ex1;
          hy = dy - lambda2 * ey2 - lambda1 * ey1;
          h12 = hx * hx + hy * hy;
          if (h12 < combined_thickness_pow2 && h12 > combined_thickness_ulim) {
            double overlap = combined_thickness - sqrt(h12);
            local_max_overlap += overlap;
            // if (overlap > local_max_overlap) local_max_overlap = overlap;
            auto result = steric_force(h12, hx, hy, lambda1, lambda2, ex1, ey1, ex2, ey2, npidx, 1, combined_thickness);
            local_forces[ptcl_idx].fx += result.fx;
            local_forces[ptcl_idx].fy += result.fy;
            local_forces[ptcl_idx].tz += result.tz;
          } else {
            lambda2 = -lambda2_prime;
            hx = dx - lambda2 * ex2 - lambda1 * ex1;
            hy = dy - lambda2 * ey2 - lambda1 * ey1;
            h12 = hx * hx + hy * hy;
            if (h12 < combined_thickness_pow2 && h12 > combined_thickness_ulim) {
              double overlap = combined_thickness - sqrt(h12);
              // if (overlap > local_max_overlap) local_max_overlap = overlap;
              local_max_overlap += overlap;
              auto result =
                  steric_force(h12, hx, hy, lambda1, lambda2, ex1, ey1, ex2, ey2, npidx, 1, combined_thickness);
              local_forces[ptcl_idx].fx += result.fx;
              local_forces[ptcl_idx].fy += result.fy;
              local_forces[ptcl_idx].tz += result.tz;
            }
          }
        }
      }
    }
  });

  // Reduction: accumulate thread-local array into global arrays
  // Forces for own particles
  for (auto& local : tls_forces) {
    for (size_t i = 0; i < ptcl.size(); ++i) {
      ptcl[i].forces += {local[i].fx, local[i].fy};
      ptcl[i].torque += local[i].tz;
    }
  }
  // Forces from neighbor interactions
  for (auto& nbor : tls_nbor_forces) {
    for (size_t i = 0; i < ptcl.size(); ++i) {
      ptcl[i].forces += {nbor[i].fx, nbor[i].fy};
      ptcl[i].torque += nbor[i].tz;
    }
  }

  // overlap between cells
  max_overlap = 0;
  for (const auto& local_max : tls_max_overlap) {
    max_overlap += local_max;
    // if (local_max > max_overlap) max_overlap = local_max;
  }
}
