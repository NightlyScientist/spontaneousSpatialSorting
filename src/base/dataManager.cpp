#include "dataManager.h"

#include <fstream>
#include <iostream>
#include <string>

#include "mdmain.h"

void read_parameters(int argc, char *argv[]) {
  if (argc < 19) {
    printf("insufficient args provided.\n");
    exit(0);
  }

  work_dir = argv[18];
  printf("work dir: %s\n", work_dir.c_str());

  cycles = (int)atoi(argv[2]);
  maxPtcls = (int)atoi(argv[3]);
  frameSize = (int)atoi(argv[4]);
  systemSize = (double)atof(argv[5]);
  printf("cycles | max ptcls | frame size | system size: %d %d %d %lf\n",
         cycles, maxPtcls, frameSize, systemSize);

  del_t = (double)atof(argv[6]);
  thickness = (double)atof(argv[7]);
  tracers = (int)atoi(argv[8]);
  printf("del_t | thickness | tracers: %lf %lf %d\n", del_t, thickness,
         tracers);

  diffusion_trans = (double)atof(argv[9]);
  forceConst = (double)atof(argv[10]);
  printf("diffusion const | force const: %lf %lf\n", diffusion_trans,
         forceConst);

  std::string _str(argv[11]);
  std::cout << "initial type is " << _str << std::endl;
  if ("uniform" == _str) {
    initial_type = 1;
  } else if ("annulus" == _str) {
    initial_type = 2;
  } else {
    initial_type = 0;
  }

  initial_cells = (int)atoi(argv[16]);
  printf(" initial cells: %d\n", initial_cells);

  double fill_fraction = (double)atof(argv[12]);
  recycleCells = (int)atoi(argv[13]);
  printf("fill fraction | recycle cells: %lf %d\n", fill_fraction,
         recycleCells);

  sscanf(argv[14], "%lf,%lf", &division_lengths[0], &division_lengths[1]);
  sscanf(argv[15], "%lf,%lf", &growth_rates[0], &growth_rates[1]);
  printf("division lengths | growth rates: %lf %lf %lf %lf\n",
         division_lengths[0], division_lengths[1], growth_rates[0],
         growth_rates[1]);

  double boxcut = 2 * division_lengths[0] + thickness;
  if (division_lengths[1] > 0.) {
    // boxcut = (division_lengths[0] + division_lengths[1]) + 2 * thickness;
    boxcut = 0.5 * (division_lengths[0] + division_lengths[1]) + 2 * thickness;
  }

  // number of spatial partitions
  lineBoxes = (int)((systemSize + 1.0) / boxcut);
  gridBoxes = lineBoxes * lineBoxes;
  scale = 1 / systemSize;

  thickness_sqrt = sqrt(thickness);
  thickness_pow2 = thickness * thickness;
  thickness_ulim = thickness * thickness * 0.25 + SMALL;

  // update force constant to be scaled by thickness
  forceConst = forceConst * thickness_sqrt;

  diffusion_rot = diffusion_trans * 2.0;
  eff_diff_rot = sqrt(del_t * 2.0 * diffusion_rot);
  eff_diff_trans = sqrt(del_t * 2.0 * diffusion_trans);

  maxArea = fill_fraction * systemSize * systemSize;
}

void save_data(int l_itime, double time) {
  int intVar;
  double x, y, z, var;

  std::string fname = work_dir + "/" + std::to_string(l_itime) + ".csv";
  std::ofstream csv(fname);

  csv << "x,y,z,ex,ey,ez,l,ancestor,color,color2,fx,fy,fz,id_1,id_2,"
         "ancestors,splits,phi,g,box_index,active,time"
      << std::endl;

  for (int i = 0; i < n_active; i++) {
    // particle position
    x = (double)ptcl[i].x * systemSize;
    y = (double)ptcl[i].y * systemSize;
    z = (double)ptcl[i].z * systemSize;

    csv << x << "," << y << "," << z << ",";
    // particle orientation
    x = (double)ptcl[i].ex;
    y = (double)ptcl[i].ey;
    z = (double)ptcl[i].ez;
    csv << x << "," << y << "," << z << ",";

    // particle length
    x = (double)ptcl[i].length;
    csv << x << ",";

    // particle ancestor
    intVar = (int)ptcl[i].ancestor;
    csv << intVar << ",";

    // particle color (label)
    intVar = (int)ptcl[i].color;
    csv << intVar << ",";

    // particle color (alt label)
    intVar = (int)ptcl[i].color2;
    csv << intVar << ",";

    // particle forces
    x = ptcl[i].force_x / ptcl[i].length;
    y = ptcl[i].force_y / ptcl[i].length;
    z = ptcl[i].force_z / ptcl[i].length;
    csv << x << "," << y << "," << z << ",";

    // particle random group assignment (?)
    intVar = (int)ptcl[i].ancestor_rnd_group;
    csv << intVar << ",";

    // particle alt group assignment (?)
    intVar = (int)ptcl[i].ind_group;
    csv << intVar << ",";

    // ancestors of ith particle
    std::string ancestors = std::to_string(ptcl[i].ancestor_list[0]);
    for (int j = 1; j < 30; j++) {
      intVar = (int)ptcl[i].ancestor_list[j];
      ancestors = ancestors + ":" + std::to_string(intVar);
    }
    csv << ancestors << ",";

    // particle division counts
    intVar = (int)ptcl[i].splits;
    csv << intVar << ",";

    // particle phi
    var = (double)ptcl[i].phi;
    csv << var << ",";

    var = (double)ptcl[i].growth_rate;
    csv << var << ",";

    // particle box index
    intVar = (int)ptcl[i].box_index;
    csv << intVar << ",";

    // particle active or not
    intVar = (int)ptcl[i].active;
    csv << intVar << ",";
    csv << time << std::endl;
  }

  csv.close();
}
