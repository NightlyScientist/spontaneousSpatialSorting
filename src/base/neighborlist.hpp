#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <array>
#include <cstddef>

#include "types/particle.hpp"
#include "types/coordinates.hpp"

using Vec64 = std::array<int, 64>;
constexpr size_t MAX_NEIGHBORS = 64;
using NeighborVec = std::array<int, MAX_NEIGHBORS>;





struct BoxAssignment {
public:
    BoxAssignment(double cell_size, double system_size, size_t nptcls);

    void updateCellBoxes(std::vector<Particle>& ptcl);
    void resizeLists(size_t new_size);

    double systemSize;
    double cellSize;
    int nBoxes;
    int nGridCells;

    std::vector<int> head;  // head of linked list per cell
    std::vector<int> list;  // linked list "next" pointers for particles
};

class NeighborList {
public:
    NeighborList(const std::vector<double>& d0, const std::vector<double>& l0, double skin);

    void build_neighbor_list(BoxAssignment& cells, std::vector<Particle>& ptcl);
    void build_neighbor_list_brute(std::vector<Particle>& ptcl);
    void setSepMatrices(const std::vector<double>& d0);

    void resizeLists(size_t nsize);

    static inline int neighborIdx(int ix, int iy, int dx, int dy, int nboxes);

    double skin;
    double cutoff[2][2];
    double cutoff_with_skin[2][2];

    Matrix m_sep;
    Matrix m_sep_sqr;
    Matrix m_sep_ulim;

    std::vector<NeighborVec> neighborlist;
    std::vector<int> neighbor_count;
};
