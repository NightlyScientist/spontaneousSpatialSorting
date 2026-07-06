#include "neighborlist.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "types/coordinates.hpp"
#include "types/particle.hpp"

inline int NeighborList::neighborIdx(int ix, int iy, int dx, int dy, int nboxes) {
  int x = ((ix + dx) % nboxes + nboxes) % nboxes;
  int y = ((iy + dy) % nboxes + nboxes) % nboxes;
  return x + nboxes * y;
}

// --- BoxAssignment Implementation ---

BoxAssignment::BoxAssignment(double cell_size_, double system_size_, size_t nptcls)
    : systemSize(system_size_), cellSize(cell_size_) {
  nBoxes = static_cast<int>(std::ceil(system_size_ / cell_size_));
  nGridCells = nBoxes * nBoxes;
  head.resize(nGridCells, -1);
  list.resize(nptcls, -1);
}

void BoxAssignment::resizeLists(size_t new_size) { list.resize(new_size, -1); }

void BoxAssignment::updateCellBoxes(std::vector<Particle>& ptcl) {
  // Reset all cell heads before assignment
  std::fill(head.begin(), head.end(), -1);

  for (size_t i = 0; i < ptcl.size(); ++i) {
    auto& p = ptcl[i];
    int ix = static_cast<int>(std::floor(p.pos[0] / cellSize));
    int iy = static_cast<int>(std::floor(p.pos[1] / cellSize));
    ix = (ix % nBoxes + nBoxes) % nBoxes;
    iy = (iy % nBoxes + nBoxes) % nBoxes;

    int flat = ix + nBoxes * iy;
    p.cell_index_flat = flat;
    list[i] = head[flat];
    head[flat] = static_cast<int>(i);
  }
}

// --- NeighborList Implementation ---

NeighborList::NeighborList(const std::vector<double>& d0, const std::vector<double>& l0, double skin_) {
  skin = skin_;
  cutoff[0][0] = l0[0];
  cutoff[1][1] = l0[1];
  cutoff[0][1] = cutoff[1][0] = 0.5 * (l0[0] + l0[1]);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) cutoff_with_skin[i][j] = cutoff[i][j] + skin;
}

void NeighborList::resizeLists(size_t nsize) {
  neighborlist.resize(nsize);
  neighbor_count.resize(nsize, 0);
}

void NeighborList::build_neighbor_list_brute(std::vector<Particle>& ptcl) {
  std::fill(neighbor_count.begin(), neighbor_count.end(), 0);

  for (size_t i = 0; i < ptcl.size(); ++i) {
    const auto& p1 = ptcl[i];
    for (size_t j = i + 1; j < ptcl.size(); ++j) {
      const auto& p2 = ptcl[j];
      int t1 = static_cast<int>(p1.type);
      int t2 = static_cast<int>(p2.type);
      double d = separation(p1.pos, p2.pos)[2];
      double rcut = cutoff_with_skin[t1][t2];
      if (d <= rcut) {
        neighborlist[i][neighbor_count[i]++] = static_cast<int>(j);
        neighborlist[j][neighbor_count[j]++] = static_cast<int>(i);
      }
    }
  }
}

// Thread-safe, no mutex, with post-symmetrization
void NeighborList::build_neighbor_list(BoxAssignment& cells, std::vector<Particle>& ptcl) {
  // Zero neighbor counts
  std::fill(neighbor_count.begin(), neighbor_count.end(), 0);

  int nboxes = cells.nBoxes;
  auto& head = cells.head;
  auto& list = cells.list;

  // Parallel: each thread only writes to its i
  tbb::parallel_for(tbb::blocked_range<size_t>(0, ptcl.size()), [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i = r.begin(); i != r.end(); ++i) {
      const auto& p1 = ptcl[i];
      int ix = p1.cell_index_flat % nboxes;
      int iy = p1.cell_index_flat / nboxes;

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          int flat = neighborIdx(ix, iy, dx, dy, nboxes);

          for (int j = head[flat]; j != -1; j = list[j]) {
            if (j <= static_cast<int>(i)) continue;  // Only add (i,j) once, i<j

            int t1 = static_cast<int>(p1.type);
            int t2 = static_cast<int>(ptcl[j].type);
            double dist = separation(p1.pos, ptcl[j].pos)[2];
            double rcut = cutoff_with_skin[t1][t2];

            if (dist <= rcut) {
              int n_i = neighbor_count[i]++;
              if (n_i < static_cast<int>(MAX_NEIGHBORS)) neighborlist[i][n_i] = j;
            }
          }
        }
      }
    }
  });

  // Now symmetrize: for each i, for each neighbor j, add i to j's list (if not present)
  // Serial, safe, cheap for moderate neighbor count
  for (size_t i = 0; i < ptcl.size(); ++i) {
    for (int n = 0; n < neighbor_count[i]; ++n) {
      int j = neighborlist[i][n];
      // Check if neighbor j already has i (safe to skip if you can tolerate duplicates)
      bool found = false;
      for (int m = 0; m < neighbor_count[j]; ++m) {
        if (neighborlist[j][m] == static_cast<int>(i)) {
          found = true;
          break;
        }
      }
      if (!found) {
        int n_j = neighbor_count[j];
        if (n_j < static_cast<int>(MAX_NEIGHBORS)) {
          neighborlist[j][n_j] = static_cast<int>(i);
          neighbor_count[j]++;
        }
      }
    }
  }
}

void NeighborList::setSepMatrices(const std::vector<double>& d0) {
  double d0_1 = d0[0];
  double d0_2 = d0[1];
  double d0_mean = (d0_1 + d0_2) / 2;

  std::array<std::array<double, 2>, 2> c_d0 = {{{d0_1, d0_mean}, {d0_mean, d0_2}}};

  std::array<std::array<double, 2>, 2> c_d0_pow2 = {
      {{std::pow(d0_1, 2), std::pow(d0_mean, 2)}, {std::pow(d0_mean, 2), std::pow(d0_2, 2)}}};

  std::array<std::array<double, 2>, 2> c_d0_ulim = c_d0_pow2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      c_d0_ulim[i][j] = c_d0_ulim[i][j] * 0.25 + SMALL;
    }
  }

  m_sep = c_d0;
  m_sep_sqr = c_d0_pow2;
  m_sep_ulim = c_d0_ulim;
}
