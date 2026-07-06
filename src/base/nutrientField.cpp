#include "nutrientField.hpp"

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <stdio.h>
#include <algorithm>

#include "types/particle.hpp"

inline void NutrientField::resetMass() {
  // zero the mass at each box of nutrient field
  std::fill(mass.begin(), mass.end(), 0);
}

void NutrientField::updateField(const std::vector<Particle>& ptcl) {
  if (disable_nutrient_field) return;

  const int n_active = static_cast<int>(ptcl.size());
  resetMass();

  // Thread-local mass accumulations
  tbb::enumerable_thread_specific<std::vector<double>> tls_mass([&] { return std::vector<double>(mass.size(), 0.0); });

  // iterate through particles and cacalculate particle(rod) area at center of rod
  tbb::parallel_for(0, n_active, [&](int ptcl_idx) {
    auto& local_mass = tls_mass.local();
    const auto& p = ptcl[ptcl_idx];
    local_mass[p.cell_index_flat] += p.thickness * p.length;
  });

  // Combine thread-local vectors into mass
  for (const auto& local_mass : tls_mass) {
    for (size_t i = 0; i < mass.size(); ++i) {
      mass[i] += local_mass[i];
    }
  }
}

bool NutrientField::updateParticles(std::vector<Particle>& ptcl) {
  if (disable_nutrient_field) return true;

  const int n_ptcl = ptcl.size();
  tbb::enumerable_thread_specific<std::vector<double>> tls_resource_deltas(
      [&] { return std::vector<double>(resources.size(), 0.0); });
  tbb::enumerable_thread_specific<int> tls_stale_count([]() { return 0; });

  // First stage: calculate changes in parallel
  tbb::parallel_for(0, n_ptcl, [&](int p_index) {
    auto& local_deltas = tls_resource_deltas.local();
    auto& p = ptcl[p_index];

    // if particle can not takeup nutrients, don't deplete them from local resources
    if (p.can_grow) {
      // calculate ratio of box mass and carrying_capacity, clamped to max 1.0
      double box_mass_ratio = std::min(mass[p.cell_index_flat] / carrying_capacity, 1.0);
      double resources_depleted = depletion_rate * box_mass_ratio * dt;
      // deplete resouces taken-up by particles
      local_deltas[p.cell_index_flat] += resources_depleted;
    }
  });

  // Serially apply resource deltas, avoiding races
  std::vector<double> global_resource_delta(resources.size(), 0.0);
  for (const auto& local_deltas : tls_resource_deltas) {
    for (size_t i = 0; i < resources.size(); ++i) {
      global_resource_delta[i] += local_deltas[i];
    }
  }
  for (size_t i = 0; i < resources.size(); ++i) {
    resources[i] -= global_resource_delta[i];
    // Clamp resources to non-negative if needed
    if (resources[i] < 0) resources[i] = 0;
  }

  // Second stage: update particles and stale count
  // update particle as being 'stale' if local resources deplete below threshold
  tbb::parallel_for(0, n_ptcl, [&](int p_index) {
    auto& local_stale_count = tls_stale_count.local();
    auto& p = ptcl[p_index];
    // Resources for this cell already updated
    const double gf = p.growth_factor = resources[p.cell_index_flat];
    p.stale = (gf <= 1e-3 || !p.can_grow);
    if (p.stale) ++local_stale_count;
  });

  int stale_count = 0;
  for (const auto& count : tls_stale_count) stale_count += count;

  if (stale_count >= n_ptcl) {
    printf("  ! all particles can longer grow, ending simulation.\n");
    return false;
  }
  return true;
}
