#include "populations.h"

#include <algorithm>
#include <iostream>

#include "interactions.h"
#include "mdmain.h"

void grow() {
  int new_active = n_active;
  int use_now = n_active;

  double o_length = 0.0;
  double l_0, growth;
  double countdown = 5000.0;
  double _growth_rate =
      std::max(division_lengths[0], division_lengths[1]) / countdown;

  double initial_lengths[n_active];
  for (int i = 0; i < n_active; i++) {
    initial_lengths[i] =
        division_lengths[ptcl[i].label] * (1 - 0.5 * drand48());
  }

  while (true) {
    periodic_boundary_condition();
    generate_linked_cell_list();

    // run in parallel
    compute_force();

    int _grown = 0;

    // iterate through ptcl list, moving and replicating
    for (int i = 0; i < n_active; i++) {
      if (ptcl[i].active == FALSE) continue;

      o_length = 1.0 / (ptcl[i].length);

      // move and scale (u,v) to the range [0,1)
      ptcl[i].x += scale * (ptcl[i].force_x * o_length * del_t);
      ptcl[i].y += scale * (ptcl[i].force_y * o_length * del_t);

      o_length = o_length * o_length * o_length;

      // update orientation
      ptcl[i].phi += 12.0 * ptcl[i].torque_z * o_length * del_t;

      // update ptcl orientation (ex, ey)
      ptcl[i].ex = cos(ptcl[i].phi);
      ptcl[i].ey = sin(ptcl[i].phi);

      // increase particle length
      growth = _growth_rate;

      if (ptcl[i].length < initial_lengths[i]) {
        ptcl[i].length += growth;
        area_particles += growth * thickness;
      } else {
        _grown += 1;
      }
    }
    if (_grown == n_active) break;
  }
}

// initialize cells with defaults
void init_cells() {
  for (int i = 0; i < maxPtcls; i++) {
    ptcl[i].active = FALSE;

    ptcl[i].x = 0.0;
    ptcl[i].y = 0.0;
    ptcl[i].z = 0.0;

    ptcl[i].ex = 0.0;
    ptcl[i].ey = 0.0;
    ptcl[i].ez = 0.0;

    ptcl[i].force_x = 0.0;
    ptcl[i].force_y = 0.0;
    ptcl[i].force_z = 0.0;

    ptcl[i].torque_x = 0.0;
    ptcl[i].torque_y = 0.0;
    ptcl[i].torque_z = 0.0;

    ptcl[i].ancestor = 0;
    ptcl[i].splits = 0;

    ptcl[i].color = -1;
    ptcl[i].color2 = -1;

    ptcl[i].label = 0;

    ptcl[i].length = -1;

    ptcl[i].growth_rate = 0.;
    ptcl[i].can_grow = true;

    for (int j = 0; j < 30; j++) {
      ptcl[i].ancestor_list[j] = 0;
    }
  }
}

// initialize with two cells at the center
int two_cells() {
  double meanLenth = 0.5 * (division_lengths[0] + division_lengths[1]);
  double radius = sqrt(thickness * meanLenth * 4);

  for (int i; i < 2; i++) {
    ptcl[i].active = TRUE;
    ptcl[i].growth_rate = growth_rates[1];
    ptcl[i].length = 0.5 * division_lengths[i];

    ptcl[i].x = (drand48() - 0.5) * radius * scale;
    ptcl[i].y = (drand48() - 0.5) * radius * scale;
    ptcl[i].z = 0;

    ptcl[i].phi = 0.0;
    ptcl[i].ex = cos(ptcl[0].phi);
    ptcl[i].ey = sin(ptcl[0].phi);

    ptcl[i].ancestor = 2 + i;
    ptcl[i].ancestor_list[0] = 1;
    ptcl[i].color = i;
    ptcl[i].color2 = i;
    ptcl[i].label = i;

    area_particles += ptcl[i].length * thickness;
  }

  remove_token = 0;
  total_anscestors = 2;
  n_active = 2;

  return 2;
}

// initialize with random cells at the center within some radius, with a ring of
// cells
int annulus(int start_idx, bool active, int label, bool grown) {
  int i = start_idx;
  double size_factor = (grown) ? 1.0 : 0.5;

  // .find extreme postions of cells
  double _radius, radius = 0;
  for (int j = 0; j < n_active; j++) {
    _radius = sqrt(pow(ptcl[j].x, 2) + pow(ptcl[j].y, 2)) * systemSize;
    if (_radius > radius) radius = _radius;
  }

  // .add half the division length to the radius
  radius = radius + thickness + .5 * std::max(division_lengths[0], division_lengths[1]);

  // .create ring around existing droplet
  double delta_angle = size_factor * division_lengths[1] / radius, angle = 0;
  int n_steps = ceil(2 * pi * radius / division_lengths[1] / size_factor);

  // .update values after adjusting radius
  radius = n_steps * size_factor * division_lengths[1] / (2 * pi);
  delta_angle =  size_factor * division_lengths[1] / radius;

  for (i = start_idx; i < start_idx + n_steps; i++) {
    angle = (i - start_idx) * delta_angle - 0.5 * delta_angle;
    double x = radius * cos(angle) * scale;
    double y = radius * sin(angle) * scale;

    ptcl[i].label = label;
    ptcl[i].length = size_factor * division_lengths[label];
    ptcl[i].growth_rate = growth_rates[label] * (1.5 - drand48());

    ptcl[i].phi = angle + pi / 2;

    ptcl[i].x = x;
    ptcl[i].y = y;
    ptcl[i].ex = cos(ptcl[i].phi);
    ptcl[i].ey = sin(ptcl[i].phi);
    ptcl[i].ancestor = i + 2;
    ptcl[i].ancestor_list[0] = i + 1;

    ptcl[i].color = i;
    ptcl[i].color2 = label;

    ptcl[i].active = true;

    if (growth_rates[label] == 0 || !active) {
      ptcl[i].can_grow = false;
    }
    n_active++;
    total_anscestors++;

    area_particles += ptcl[i].length * thickness;
  }

  remove_token = 0;
  return i;
}

int random_cells(int start_idx, int Start_parts, int _label = -1) {
  int i = start_idx;

  // make start_parts even
  if (Start_parts % 2 == 1) Start_parts += 1;

  double meanLenth = 0.5 * (division_lengths[0] + division_lengths[1]);
  double radius = sqrt(thickness * meanLenth * Start_parts);

  for (i = start_idx; i < Start_parts + start_idx; i++) {
    // set label based on even/odd
    int label = i % 2;
    if (_label == -1) label = 0;

    ptcl[i].label = label;
    ptcl[i].length = 0.01;
    ptcl[i].growth_rate = growth_rates[label] * (1.5 - drand48());

    ptcl[i].phi = drand48() * 2 * pi;

    ptcl[i].x = (drand48() - 0.5) * radius * scale;
    ptcl[i].y = (drand48() - 0.5) * radius * scale;
    ptcl[i].ex = cos(ptcl[i].phi);
    ptcl[i].ey = sin(ptcl[i].phi);
    ptcl[i].ancestor = i + 2;
    ptcl[i].ancestor_list[0] = i + 1;

    ptcl[i].color = i;
    ptcl[i].color2 = label;

    ptcl[i].active = TRUE;

    if (growth_rates[label] == 0) {
      ptcl[i].can_grow = false;
    }
    n_active++;
    total_anscestors++;

    area_particles += ptcl[i].length * thickness;
  }

  remove_token = 0;
  return i;
}

int tracer_cells(int start_idx) {
  int i = start_idx;
  for (int j = start_idx; j < tracers + start_idx; j++) {
    ptcl[j].color = i;
    ptcl[j].active = TRUE;
    ptcl[j].x = 2.0 * (drand48() - 0.5) * (0.5 - 3.0 * thickness * scale);
    ptcl[j].y = 2.0 * (drand48() - 0.5) * (0.5 - 2.0 * thickness * scale);
    ptcl[j].phi = 0.0;
    ptcl[j].ex = cos(ptcl[i].phi);
    ptcl[j].ey = sin(ptcl[i].phi);
    ptcl[j].length = thickness;
    ptcl[j].growth_rate = 0.0;
    ptcl[j].color2 = 2;
    n_active++;
  }
  return i;
}

void starting_cells() {
  init_cells();
  int start_index = 0;
  switch (initial_type) {
    case 0:
      two_cells();
      break;

    case 1:
      random_cells(0, initial_cells, 1);
      grow();
      break;

    case 2:
      start_index = random_cells(start_index, initial_cells);
      grow();
      annulus(start_index);
      break;

    case 3:
      random_cells(0, initial_cells, -1);
      grow();
      break;

    default:
      break;
  }

  periodic_boundary_condition();
  generate_linked_cell_list();
}
