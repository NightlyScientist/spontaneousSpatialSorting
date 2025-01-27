#ifndef FORCE
#define FORCE

void distance(double l_x1, double l_y1, double l_z1, double l_x2, double l_y2,
              double l_z2, double *l_dx, double *l_dy, double *l_dz,
              double *l_r_2);

void compute_force();

int neighborIdx(int ix, int iy, int jx, int jy);

void steric_force(double h12, double hx, double hy, double lambda1,
                  double lambda2, double dx, double dy, double ex1, double ey1,
                  double ex2, double ey2, int actuaparticle,
                  int neighbor_particle, int interaction_type);

void generate_linked_cell_list(void);
void periodic_boundary_condition(void);
void uv_xyz(double u, double v, double *x, double *y, double *z);

#endif /*force*/
