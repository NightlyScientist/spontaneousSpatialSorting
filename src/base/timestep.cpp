#include "timestep.h"

#include "dataManager.h"
#include "interactions.h"
#include "mdmain.h"
#include "populations.h"

double exp_dist(double diff_t) {
  double rand_num = drand48();
  return -diff_t * log(rand_num);
}

double gauss(double sigma, double mu) {
  double U1 = drand48();
  double U2 = drand48();
  return sqrt(-2.0 * log(U1)) * cos(2. * pi * U2) * sigma + mu;
}

int timeStep() {
  int new_active = n_active;
  int use_now = n_active;
  int label;

  double diff_x = 0.0, diff_y = 0.0;
  double rnd_phi = 0.0, o_length = 0.0;
  double l_0, growth;

  // run in parallel
  compute_force();

  // iterate through ptcl list, moving and replicating
  for (int i = 0; i < n_active; i++) {
    if (ptcl[i].active == FALSE) continue;

    if (eff_diff_trans != 0.0) {
      diff_x = eff_diff_trans * gauss(1, 0.0);
      diff_y = eff_diff_trans * gauss(1, 0.0);
    }

    o_length = 1.0 / (ptcl[i].length);

    ptcl[i].x +=
        scale * (forceConst * ptcl[i].force_x * o_length * del_t + diff_x);
    ptcl[i].y +=
        scale * (forceConst * ptcl[i].force_y * o_length * del_t + diff_y);

    o_length = o_length * o_length * o_length;

    // update orientation
    if (eff_diff_rot != 0.0) {
      rnd_phi = eff_diff_rot * gauss(1.0, 0.0);
    }
    ptcl[i].phi +=
        forceConst * 12.0 * ptcl[i].torque_z * o_length * del_t + rnd_phi;

    // update ptcl orientation (ex, ey)
    ptcl[i].ex = cos(ptcl[i].phi);
    ptcl[i].ey = sin(ptcl[i].phi);

    // increase particle length
    if (ptcl[i].can_grow) {
      growth = ptcl[i].growth_rate * del_t;
      label = ptcl[i].label;

      ptcl[i].length += growth;
      area_particles += growth * thickness;
    }

    if (ptcl[i].can_grow && (label < 2) &&
        (ptcl[i].length >= division_lengths[label])) {
      if ((recycleCells == 2) && (remove_token == 1) && (drand48() < 0.5)) {
        ptcl[i].active = FALSE;
        removed_list[N_removed] = i;
        N_removed++;
      } else {
        if (N_removed > 0) {
          use_now = removed_list[N_removed - 1];
          N_removed--;
        } else {
          use_now = new_active;
          new_active++;
        }

        // add ptcl to active list
        ptcl[use_now].active = TRUE;

        // update (u,v) and (x,y,z)
        ptcl[use_now].x =
            ptcl[i].x + (0.25 * ptcl[i].length) * ptcl[i].ex * scale;
        ptcl[use_now].y =
            ptcl[i].y + (0.25 * ptcl[i].length) * ptcl[i].ey * scale;

        // inherit initial length from ancestor
        ptcl[use_now].length = 0.5 * ptcl[i].length;

        // investigate: is this necessary:
        ptcl[use_now].x -= rint(ptcl[use_now].x);
        ptcl[use_now].y -= rint(ptcl[use_now].y);

        // inherit orientation
        ptcl[use_now].ex = ptcl[i].ex;
        ptcl[use_now].ey = ptcl[i].ey;
        ptcl[use_now].phi = ptcl[i].phi;

        // inherit labels
        ptcl[use_now].color = ptcl[i].color;
        ptcl[use_now].color2 = ptcl[i].color2;
        ptcl[use_now].label = label;

        ptcl[use_now].growth_rate = growth_rates[label] * (1.5 - drand48());

        if (i == 0) {
          ind_group_number++;
          ptcl[use_now].ind_group = ind_group_number;
        } else
          ptcl[use_now].ind_group = ptcl[i].ind_group;

        int nm = 0;
        while (ptcl[i].ancestor_list[nm] > 0) {
          ptcl[use_now].ancestor_list[nm] = ptcl[i].ancestor_list[nm];
          nm++;
        }
        ptcl[i].ancestor_list[nm] = ptcl[i].ancestor;
        ptcl[use_now].ancestor_list[nm] = ptcl[i].ancestor;

        ptcl[i].ancestor = total_anscestors;
        ptcl[use_now].ancestor = total_anscestors + 1;
        total_anscestors = total_anscestors + 2;

        if (use_now < ancestor_start_count)
          ptcl[use_now].ancestor_rnd_group = use_now % 2;
        else
          ptcl[use_now].ancestor_rnd_group = ptcl[i].ancestor_rnd_group;

        // update (u,v) and (x,y,z)
        ptcl[i].x -= (0.25 * ptcl[i].length) * ptcl[i].ex * scale;
        ptcl[i].y -= (0.25 * ptcl[i].length) * ptcl[i].ey * scale;

        // update ancestor cell length
        ptcl[i].length = 0.5 * ptcl[i].length;

        // investigate: is this necessary:
        ptcl[i].x -= rint(ptcl[i].x);
        ptcl[i].y -= rint(ptcl[i].y);
        // recrod number of cell divisions
        ptcl[i].splits += 1;

        if (new_active == maxPtcls) {
          printf("  ! particle amount exceeded, ending simulation.\n");
          n_active = new_active;
          return 1;
        }
      }
    }
  }

  n_active = new_active;
  if (area_particles > maxArea) {
    if (recycleCells != 0)
      remove_token = 1;
    else {
      printf("  ! reached filling fraction, ending simulation.\n");
      return 1;
    }
  }
  return 0;
}
