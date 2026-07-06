#pragma once

#include <vector>
#include "coordinates.hpp"

//------------------- Settings, types --------------------//
constexpr int NIDS = 3;
constexpr int NUM_TYPES = 2;
enum class ParticleType : int { A = 0, B = 1, P = 2};
constexpr double pi = 3.14159265358979323846;
constexpr double SMALL = 10e-16;

//------------------ Particle definition -------------------
struct Particle {
  Vec2 pos = {0,0};  // Center position
  Vec2 ori = {0,0};  // Unit direction vector
  Vec2 pos_last_update = {0,0};  // For skin logic
  std::vector<int> neighbors; // Cell neighbors
  double length = 0; // particle length
  ParticleType type; // rod type (bi-disperse)
  Vec2 forces = {0,0};
  double torque = 0;
  double growth_rate = 0.0001;
  double phi = 0;
  int ancestor = 0;
  int splits = 0;
  int ancestor_rnd_group = 0;
  std::vector<int> ancestor_list;
  int cell_index_1 = -1, cell_index_2 = -1, cell_index_flat = -1;
  bool can_grow = true, stale = false;
  double growth_factor = 1.0;
  double thickness = 0.1;
};

struct DistanceResult {
  double dx, dy, r;
};

inline DistanceResult center_dist(const Particle& ri, const Particle& rj, const double box) {
  Vec2 dr = pbc_vec(ri.pos, rj.pos, box);
  return {dr[0], dr[1], norm(dr)};
}
