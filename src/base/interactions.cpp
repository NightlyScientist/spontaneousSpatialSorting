#include "interactions.h"

#include <math.h>

#include "mdmain.h"
#include "populations.h"
#include "timestep.h"

void generate_linked_cell_list() {
  for (int i = 0; i < gridBoxes; i++) head[i] = -1;

  int boxindex, ibox, jbox;
  for (int i = 0; i < n_active; i++) {
    ibox = (int)floor(ptcl[i].x * lineBoxes + 0.5 * lineBoxes);
    jbox = (int)floor(ptcl[i].y * lineBoxes + 0.5 * lineBoxes);

    ptcl[i].idx_1 = ibox;
    ptcl[i].idx_2 = jbox;

    boxindex = ibox + jbox * lineBoxes;
    ptcl[i].box_index = boxindex;

    list[i] = head[boxindex];
    head[boxindex] = i;
  }
}

void periodic_boundary_condition() {
  int i = 0;
  for (i = 0; i < n_active; i++) {
    ptcl[i].x -= rint(ptcl[i].x);
    ptcl[i].y -= rint(ptcl[i].y);
    ptcl[i].z -= rint(ptcl[i].z);
  }
}

void distance(double x1, double y1, double x2, double y2, double *dx,
              double *dy, double *r_2) {
  *dx = x1 - x2;
  *dy = y1 - y2;

  *dx = *dx - rint(*dx);
  *dy = *dy - rint(*dy);

  *dx = *dx * systemSize;
  *dy = *dy * systemSize;

  *r_2 = *dx * *dx + *dy * *dy;
}

int neighborIdx(int ix, int iy, int jx, int jy) {
  return head[(ix + jx + lineBoxes) % lineBoxes +
              lineBoxes * ((iy + jy + lineBoxes) % lineBoxes)];
}

void compute_force() {
  for (int ptcl_idx = 0; ptcl_idx < n_active; ptcl_idx++) {
    ptcl[ptcl_idx].force_x = 0.0;
    ptcl[ptcl_idx].force_y = 0.0;
    ptcl[ptcl_idx].torque_z = 0.0;
  }

#pragma omp parallel for shared(head, ptcl, thickness, thickness_pow2, \
                                    thickness_sqrt, thickness_ulim)
  for (int ptcl_idx = 0; ptcl_idx < n_active; ptcl_idx++) {
    double dx, dy;
    double hx, hy, h12;
    double lambda1, lambda2;
    double rmin_2;
    double x1, y1, ex1, ey1;
    double x2, y2, ex2, ey2;
    int ix, iy, nbor_ptcl_idx, orientation_sign;

    // position is based on s(u,v) for the torus
    x1 = ptcl[ptcl_idx].x;
    y1 = ptcl[ptcl_idx].y;

    // orientation in in the local frame at (x,y)
    ex1 = ptcl[ptcl_idx].ex;
    ey1 = ptcl[ptcl_idx].ey;

    ix = ptcl[ptcl_idx].idx_1;
    iy = ptcl[ptcl_idx].idx_2;

    nbor_ptcl_idx = -1;

    for (int jx = -1; jx < 2; jx++) {
      for (int jy = -1; jy < 2; jy++) {
        nbor_ptcl_idx = neighborIdx(ix, iy, jx, jy);

        while (nbor_ptcl_idx != -1) {
          if (nbor_ptcl_idx != ptcl_idx && ptcl[nbor_ptcl_idx].active == TRUE &&
              ptcl[ptcl_idx].active == TRUE) {
            // neighbr position
            x2 = ptcl[nbor_ptcl_idx].x;
            y2 = ptcl[nbor_ptcl_idx].y;

            // neighrbor orientation
            ex2 = ptcl[nbor_ptcl_idx].ex;
            ey2 = ptcl[nbor_ptcl_idx].ey;

            // distance between pair of particles
            distance(x1, y1, x2, y2, &dx, &dy, &rmin_2);

            if (rmin_2 <=
                pow(0.5 * (ptcl[ptcl_idx].length + ptcl[nbor_ptcl_idx].length),
                    2)) {
              // check confirugations for interactions
              orientation_sign = 0;

              while (orientation_sign < 2) {
                lambda1 = (ptcl[ptcl_idx].length - thickness) * 0.5;

                if (orientation_sign == 1) lambda1 = -lambda1;

                lambda2 =
                    (dx - lambda1 * ex1) * ex2 + (dy - lambda1 * ey1) * ey2;

                hx = dx - lambda2 * ex2 - lambda1 * ex1;
                hy = dy - lambda2 * ey2 - lambda1 * ey1;
                h12 = hx * hx + hy * hy;

                double lambda2_prime =
                    (ptcl[nbor_ptcl_idx].length - thickness) * 0.5;

                if (h12 < thickness_pow2 && h12 > thickness_ulim &&
                    lambda2 * lambda2 < lambda2_prime * lambda2_prime) {
                  steric_force(h12, hx, hy, lambda1, lambda2, dx, dy, ex1, ey1,
                               ex2, ey2, ptcl_idx, nbor_ptcl_idx, 0);
                } else {
                  lambda2 = lambda2_prime;
                  hx = dx - lambda2 * ex2 - lambda1 * ex1;
                  hy = dy - lambda2 * ey2 - lambda1 * ey1;
                  h12 = hx * hx + hy * hy;

                  if (h12 < thickness_pow2 && h12 > thickness_ulim) {
                    steric_force(h12, hx, hy, lambda1, lambda2, dx, dy, ex1,
                                 ey1, ex2, ey2, ptcl_idx, nbor_ptcl_idx, 1);
                  } else {
                    lambda2 = -lambda2_prime;
                    hx = dx - lambda2 * ex2 - lambda1 * ex1;
                    hy = dy - lambda2 * ey2 - lambda1 * ey1;
                    h12 = hx * hx + hy * hy;
                    if (h12 < thickness_pow2 && h12 > thickness_ulim) {
                      steric_force(h12, hx, hy, lambda1, lambda2, dx, dy, ex1,
                                   ey1, ex2, ey2, ptcl_idx, nbor_ptcl_idx, 1);
                    }
                  }
                }
                orientation_sign++;
              }
            }
          }

          if (nbor_ptcl_idx < n_active)
            nbor_ptcl_idx = list[nbor_ptcl_idx];
          else
            nbor_ptcl_idx = -1;
        }
      }
    }
  }
}

void steric_force(double h12, double hx, double hy, double lambda1,
                  double lambda2, double dx, double dy, double ex1, double ey1,
                  double ex2, double ey2, int ptcl_idx, int nbor_ptcl_idx,
                  int interaction_type) {
  double cx1 = -lambda1 * ex1 + hx * 0.5;
  double cy1 = -lambda1 * ey1 + hy * 0.5;

  double cx2 = -lambda2 * ex2 - hx * 0.5;
  double cy2 = -lambda2 * ey2 - hy * 0.5;

  h12 = sqrt(h12);
  hx = hx / h12;
  hy = hy / h12;
  h12 = thickness - h12;
  h12 = h12 * h12 * h12;
  double fi_str = sqrt(h12);
  // double fi_str = forceConst * thickness_sqrt * sqrt(h12);

#pragma omp reduction(+ : ptcl[ptcl_idx].force_x)
  ptcl[ptcl_idx].force_x += fi_str * hx;
#pragma omp reduction(+ : ptcl[ptcl_idx].force_y)
  ptcl[ptcl_idx].force_y += fi_str * hy;
#pragma omp reduction(+ : ptcl[ptcl_idx].torque_z)
  // ptcl[ptcl_idx].torque_z += cx1 * fi_str * hy - cy1 * fi_str * hx;
  ptcl[ptcl_idx].torque_z += fi_str * (cx1 * hy - cy1 * hx);

  if (interaction_type == 0) {
#pragma omp reduction(+ : ptcl[ptcl_idx].torque_Z)
    // ptcl[nbor_ptcl_idx].torque_z += fi_str * cx2 * hy - fi_str * cy2 * hx;
    ptcl[nbor_ptcl_idx].torque_z += fi_str * (cx2 * hy - cy2 * hx);
#pragma omp reduction(+ : ptcl[ptcl_idx].force_x)
    ptcl[nbor_ptcl_idx].force_x -= fi_str * hx;
#pragma omp reduction(+ : ptcl[ptcl_idx].force_y)
    ptcl[nbor_ptcl_idx].force_y -= fi_str * hy;
  }
}
