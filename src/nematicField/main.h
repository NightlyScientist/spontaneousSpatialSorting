#ifndef MDMAIN
#define MDMAIN

// #ifndef DEBUG
//     #define assert //
// #endif /*DEBUG*/

// #define BINARYOUTPUT

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/times.h>
#include <assert.h>
#include <setjmp.h>

#define TRUE 1
#define FALSE 0

#define neighbor_max 50000
#define pi 3.141592654
#define box_list_len 10000
#define max_number_correlation_points 10000
#define SMALL 10e-16

struct molecule_var
{
    double x, y, z;
    double ex, ey, ez;
    double ex_p, ey_p, ez_p;
    double force_x, force_y, force_z;
    double torque_x, torque_y, torque_z;
    int b_i, b_j, b_k;
    double length;
    double growth_rate;
    int active;
    int ancestor;
    double phi;
};

struct nematic_field_var
{
    double x, y;
    double Q11, Q12;
    double rho;
    double vx, vy;
    double press;
};

extern struct molecule_var *molecule;

extern int N_particle;
extern char work_dir[400];

#endif /*MDMAIN*/
