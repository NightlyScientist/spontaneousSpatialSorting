#pragma once

#include <math.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "SimulationContext.h"
#include "types/coordinates.hpp"
#include "types/particle.hpp"

double seg_seg_dist(const Vec2& p1, const Vec2& q1, const Vec2& p2, const Vec2& q2);
bool capsules_overlap(const Particle& a, const Particle& b);

//------------------ Grid/Spatial Partitioning --------------
struct Grid2D {
  int nx, ny;
  std::vector<std::vector<int>> cells;
  double width, height;
  double cell_size;
  Grid2D(double w, double h, double cell) : width(w), height(h), cell_size(cell) {
    nx = std::ceil(width / cell_size);
    ny = std::ceil(height / cell_size);
    cells.assign(nx * ny, {});
  }
  int index(int ix, int iy) const { return iy * nx + ix; }
  std::vector<int> covered_cells(const Particle& c) const {
    Vec2 d = c.ori;
    double hw = 0.5 * c.length * std::abs(d[0]);
    double hh = 0.5 * c.length * std::abs(d[1]);
    double r = c.thickness / 2;
    double x0 = c.pos[0] - hw - r, x1 = c.pos[0] + hw + r;
    double y0 = c.pos[1] - hh - r, y1 = c.pos[1] + hh + r;
    int i0 = std::clamp(int(x0 / cell_size), 0, nx - 1), i1 = std::clamp(int(x1 / cell_size), 0, nx - 1);
    int j0 = std::clamp(int(y0 / cell_size), 0, ny - 1), j1 = std::clamp(int(y1 / cell_size), 0, ny - 1);
    std::vector<int> out;
    for (int ix = i0; ix <= i1; ++ix)
      for (int iy = j0; iy <= j1; ++iy) out.push_back(index(ix, iy));
    return out;
  }
  void insert(const Particle& c, int idx) {
    for (int cell : covered_cells(c)) cells[cell].push_back(idx);
  }
  void reset() {
    for (auto& cell : cells) cell.clear();
  }
};

void place_uniform_particles(std::vector<Particle>& particles, Grid2D& grid, int n, ParticleType type, double thickness,
                             double length, double box_x, double box_y, const std::string& region_type = "box",
                             double r1 = 0, double r2 = 0, int base_seed = 1);
std::vector<Particle> generate_disk_ring_tracer(int n_disk, double thick_disk, double len_disk, double growth_rate_A,
                                                int n_ring, double thick_ring, double len_ring, double growth_rate_B,
                                                int n_tracer, double thick_trace, double len_trace, double box_x,
                                                double box_y, double r_disk, double r_ring_inner, double r_ring_outer,
                                                int base_seed);
std::tuple<double, double, double, double> calc_disk_ring_dimensions(int nA, double lA, double wA, int nB, double lB,
                                                                     double wB, double disk_packing_fraction = 0.5,
                                                                     double delta = 1.0);
std::tuple<double, double> calc_disk_radius_for_mixture(int nA, double lA, double wA, int nB, double lB, double wB,
                                                        double packing_fraction = 0.5);
std::vector<Particle> disk_mixture(int nA, int nB, double wA, double wB, double lA, double lB, int base_seed = 1);
std::vector<Particle> disk_ring(int nA, int nB, double wA, double wB, double lA, double lB, int base_seed = 1);
void assign_parameters(std::vector<Particle>& particles, const SimulationParameters prms);
void shift_coordinates(std::vector<Particle>& particles, const SimulationParameters prms);
std::vector<Particle> starting_cells(const SimulationParameters& ctx, const int base_seed);
