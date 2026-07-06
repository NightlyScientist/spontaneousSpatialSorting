#include "populations.hpp"

#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "SimulationContext.h"
#include "types/coordinates.hpp"
#include "types/particle.hpp"

double seg_seg_dist(const Vec2& p1, const Vec2& q1, const Vec2& p2, const Vec2& q2) {
  Vec2 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
  double s = 0, t = 0;
  if (a <= 1e-12 && e <= 1e-12) return norm(p1 - p2);
  if (a <= 1e-12) {
    s = 0;
    t = std::clamp(f / e, 0.0, 1.0);
  } else {
    double c = dot(d1, r);
    if (e <= 1e-12) {
      t = 0;
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      double b = dot(d1, d2), denom = a * e - b * b;
      s = denom != 0 ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0;
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1) {
        t = 1;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }
  Vec2 cp1 = p1 + s * d1, cp2 = p2 + t * d2;
  return norm(cp1 - cp2);
}

//------------------ Capsule collision test -----------------
bool capsules_overlap(const Particle& a, const Particle& b) {
  Vec2 a0 = a.pos - 0.5 * a.length * a.ori, a1 = a.pos + 0.5 * a.length * a.ori;
  Vec2 b0 = b.pos - 0.5 * b.length * b.ori, b1 = b.pos + 0.5 * b.length * b.ori;
  double min_dist = seg_seg_dist(a0, a1, b0, b1);
  return min_dist < 0.5 * (a.thickness + b.thickness);
}

//------------------ Uniform rod placement, particle version -------------
void place_uniform_particles(std::vector<Particle>& particles, Grid2D& grid, int n, ParticleType type, double thickness,
                             double length, double box_x, double box_y, const std::string& region_type, double r1,
                             double r2, int base_seed) {
  std::mt19937 gen(base_seed);
  std::uniform_real_distribution<double> dir_angdis(0, 2 * M_PI);

  Vec2 center = {box_x / 2, box_y / 2};

  auto rand_dir = [&] {
    double a = dir_angdis(gen);
    return Vec2{std::cos(a), std::sin(a)};
  };

  auto rand_pos = [&] {
    if (region_type == "box") {
      std::uniform_real_distribution<double> xdis(0.5 * thickness, box_x - 0.5 * thickness);
      std::uniform_real_distribution<double> ydis(0.5 * thickness, box_y - 0.5 * thickness);
      return Vec2{xdis(gen), ydis(gen)};
    } else if (region_type == "disk") {
      std::uniform_real_distribution<double> angdis(0, 2 * M_PI);
      std::uniform_real_distribution<double> rdis(0, r1);
      double r = std::sqrt(rdis(gen));
      double ang = angdis(gen);
      return Vec2{center[0] + r * std::cos(ang), center[1] + r * std::sin(ang)};
    } else if (region_type == "annulus") {
      std::uniform_real_distribution<double> angdis(0, 2 * M_PI);
      std::uniform_real_distribution<double> rdis(r1, r2);
      double r = std::sqrt(rdis(gen) * rdis(gen));
      double ang = angdis(gen);
      return Vec2{center[0] + r * std::cos(ang), center[1] + r * std::sin(ang)};
    } else
      throw std::runtime_error("Unknown region_type for rod placement");
  };

  int max_trials_per = 10000;
  for (int i = 0; i < n; ++i) {
    bool got = false;
    for (int trial = 0; trial < max_trials_per; ++trial) {
      Vec2 pos = rand_pos();
      Vec2 dir = rand_dir();
      Particle p;
      p.ori = dir;
      p.length = length;
      p.thickness = thickness;
      p.type = type;
      p.pos = pos;

      auto near_cells = grid.covered_cells(p);
      bool overlap = false;
      for (int cell : near_cells)
        for (int idx : grid.cells[cell])
          if (capsules_overlap(p, particles[idx])) {
            overlap = true;
            break;
          }
      if (!overlap) {
        int idx = particles.size();
        particles.push_back(p);
        grid.insert(p, idx);
        got = true;
        break;
      }
    }
    if (!got) throw std::runtime_error("Failed to place rod after many trials.");
  }
}

//------------------ Disk+Ring+Tracer demo setup -------------------------
std::vector<Particle> generate_disk_ring_tracer(int n_disk, double thick_disk, double len_disk, double growth_rate_A,
                                                int n_ring, double thick_ring, double len_ring, double growth_rate_B,
                                                int n_tracer, double thick_trace, double len_trace, double box_x,
                                                double box_y, double r_disk, double r_ring_inner, double r_ring_outer,
                                                int base_seed) {
  double max_thick = std::max({thick_disk, thick_ring, thick_trace});
  double max_len = std::max({len_disk, len_ring, len_trace});
  double cell_size = max_len + max_thick;
  Grid2D grid(box_x, box_y, cell_size);
  std::vector<Particle> particles;

  // Disk type A
  if (n_disk > 0)
    place_uniform_particles(particles, grid, n_disk, ParticleType::A, thick_disk, len_disk, box_x, box_y, "disk",
                            r_disk, 0, base_seed);

  // Annulus type B
  if (n_ring > 0)
    place_uniform_particles(particles, grid, n_ring, ParticleType::B, thick_ring, len_ring, box_x, box_y, "annulus",
                            r_ring_inner, r_ring_outer, base_seed);

  // Tracers: uniform in box
  if (n_tracer > 0)
    place_uniform_particles(particles, grid, n_tracer, ParticleType::P, thick_trace, len_trace, box_x, box_y, "box",
                            base_seed);

  return particles;
}

/**
 * @brief Compute automatic region boundaries for disk and ring arrangements.
 * @param nA   Number of type A rods (disk)
 * @param lA   Type A rod length
 * @param wA   Type A rod thickness (full width, diameter)
 * @param nB   Number of type B rods (ring)
 * @param lB   Type B rod length
 * @param wB   Type B rod thickness
 * @param disk_packing_fraction  Area fraction in disk for type A (suggest 0.4~0.6)
 * @param delta  Extra buffer (in units of length), e.g., 1.0
 * @return (r_disk, r_margin, r_ring_inner, r_ring_outer)
 */
std::tuple<double, double, double, double> calc_disk_ring_dimensions(int nA, double lA, double wA, int nB, double lB,
                                                                     double wB, double disk_packing_fraction,
                                                                     double delta) {
  // 1. Compute area to enclose given #rods at the desired packing fraction.
  // Approximate each rod as lA*wA rectangle.
  double area_rods_A = nA * (lA * wA);
  double disk_area = area_rods_A / disk_packing_fraction;
  double r_disk = std::sqrt(disk_area / M_PI);

  // 2. Disk region for A rod centers: avoid poking out
  double rA = 0.5 * wA;
  double r_disk_for_centers = r_disk - (lA / 2.0 + rA);

  double rB = 0.5 * wB;
  // 3. Margin for safe gap between disk and ring
  double r_margin = (lA / 2.0 + rA) + (lB / 2.0 + rB) + delta;
  double r_ring_inner = r_disk + r_margin;
  double r_ring_outer = r_ring_inner + lB + 2 * rB;

  return std::make_tuple(r_disk_for_centers, r_margin, r_ring_inner, r_ring_outer);
}

// Calculate a safe "disk radius" for the desired rods and packing:
std::tuple<double, double> calc_disk_radius_for_mixture(int nA, double lA, double wA, int nB, double lB, double wB,
                                                        double packing_fraction) {
  double areaA = nA * lA * wA;
  double areaB = nB * lB * wB;
  double total_area = areaA + areaB;
  double disk_area = total_area / packing_fraction;
  double r_disk = std::sqrt(disk_area / M_PI);
  // For rod centers, avoid poking out:
  double max_extent = std::max(lA / 2.0 + 0.5 * wA, lB / 2.0 + 0.5 * wB);
  return std::make_tuple(r_disk, r_disk - max_extent);
}

std::vector<Particle> disk_mixture(int nA, int nB, double wA, double wB, double lA, double lB, int base_seed) {
  double phi = 0.3;
  double r_disk, r_centers;

  std::tie(r_disk, r_centers) = calc_disk_radius_for_mixture(nA, lA, wA, nB, lB, wB, phi);
  double box_x = 2.1 * r_disk, box_y = 2.1 * r_disk;

  double cell_size = std::max(lA, wA) + std::max(lB, wB);
  Grid2D grid(box_x, box_y, cell_size);
  std::vector<Particle> ps;
  // Place uniformly in same disk:
  place_uniform_particles(ps, grid, nA, ParticleType::A, wA, lA, box_x, box_y, "disk", r_centers, base_seed);
  place_uniform_particles(ps, grid, nB, ParticleType::B, wB, lB, box_x, box_y, "disk", r_centers, base_seed);
  return ps;
}

std::vector<Particle> disk_ring(int nA, int nB, double wA, double wB, double lA, double lB, int base_seed) {
  double packing_fraction = 0.5;
  double delta = 1.2;

  // Automatically determine safe radii for disk and ring:
  double r_disk, r_margin, r_ring_inner, r_ring_outer;
  std::tie(r_disk, r_margin, r_ring_inner, r_ring_outer) =
      calc_disk_ring_dimensions(nA, lA, wA, nB, lB, wB, packing_fraction, delta);

  double box_x = 2.1 * r_ring_outer, box_y = 2.1 * r_ring_outer;
  std::vector<Particle> ps;

  double cell_size = std::max({lA, wA, lB, wB});
  Grid2D grid(box_x, box_y, cell_size);

  place_uniform_particles(ps, grid, nA, ParticleType::A, wA, lA, box_x, box_y, "disk", r_disk,
                          base_seed);  // type A in disk
  place_uniform_particles(ps, grid, nB, ParticleType::B, wB, lB, box_x, box_y, "annulus", r_ring_inner, r_ring_outer,
                          base_seed);  // type B in ring
  return ps;
}

void assign_parameters(std::vector<Particle>& particles, const SimulationParameters prms) {
  for (size_t i = 0; i < particles.size(); ++i) {
    auto& p = particles[i];
    p.growth_rate = prms.growth_rates[static_cast<int>(p.type)];
    p.pos_last_update = p.pos;
    p.thickness = prms.thickness_arr[static_cast<int>(p.type)];
    p.can_grow = (p.type != ParticleType::P && p.growth_rate > 0.0) ? true : false;
    p.ancestor_rnd_group = i;
    p.ancestor = i;
    p.phi = 2 * M_PI * drand48();
  }
}

void shift_coordinates(std::vector<Particle>& particles, const SimulationParameters prms) {
  double c_x = prms.systemSize / 2;
  double c_y = c_x;
  for (auto& p : particles) {
    p.pos += {c_x, c_y};
  }
}

std::vector<Particle> starting_cells(const SimulationParameters& ctx, const int base_seed) {
  const auto d0 = ctx.thickness_arr;
  const auto l = ctx.division_lengths;
  const auto gr = ctx.growth_rates;

  int n = ctx.initial_cells;
  int nA, nB, nTracer = ctx.tracers;

  double wT, lT;
  double grA = gr[0], grB = gr[1];
  // double lA = l[0] / 2, wA = d0[0], lB = l[1] / 2, wB = d0[1];
  double lA = d0[0], wA = d0[0], lB = d0[1], wB = d0[1];
  double disk_packing_fraction = 0.5;
  double delta = 1.5;
  double r_disk, r_margin, r_ring_inner, r_ring_outer;

  std::vector<Particle> particles;
  if (ctx.initial_type == "uniform") {
    // uniform distribution
    nA = nB = int(n / 2);
    particles = disk_mixture(nA, nB, wA, wB, lA, lB, base_seed);
  } else if (ctx.initial_type == "uniform_ring") {
    // uniform distribution with a ring around
    nA = n;
    nB = nTracer;
    particles = disk_ring(nA, nB, wA, wB, lA, lB, base_seed);
  } else if (ctx.initial_type == "passive_ring") {
    // passive ring distribution
    nA = n;
    nB = nTracer;
    particles = disk_ring(nA, nB, wA, wB, lA, lB, base_seed);
  } else {
    std::cout << "could not interpret initial conditions";
    exit(1);
  }
  assign_parameters(particles, ctx);
  shift_coordinates(particles, ctx);
  return particles;
}
