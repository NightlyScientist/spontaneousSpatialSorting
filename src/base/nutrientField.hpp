#pragma once

#include <vector>

#include "types/particle.hpp"

struct NutrientField {
  NutrientField(const int nBoxes, const double capacity, double rate, double dt_)
      : depletion_rate(rate), carrying_capacity(capacity), dt(dt_) {
    const int total_boxes = nBoxes * nBoxes;
    resources.resize(total_boxes, 1.0);
    mass.resize(total_boxes, 0.0);
    if (rate == 0.0 || capacity < 0.001) disable_nutrient_field = true;
  }

  void updateField(const std::vector<Particle>& ptcl);
  bool updateParticles(std::vector<Particle>& ptcl);
  inline void resetMass();

  const double depletion_rate;
  const double carrying_capacity;
  const double dt;
  std::vector<double> resources;  // a.k.a growth factor
  std::vector<double> mass;

 private:
  bool disable_nutrient_field = false;
};
