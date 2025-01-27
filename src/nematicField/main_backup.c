#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#include <setjmp.h>
// #include <omp.h>

#include "main.h"
#define BINARYOUTPUT

struct molecule_var *molecule;
struct nematic_field_var *nematic_field;

int N_particle;
int iter_num;
int save_step;
int start_num;
char work_dir[200];

double thickness;
int grid;
double sigma;
double W_const;
double max;
double o_max;

int main(int argc, char *argv[])
{

    // cli options
    sprintf(work_dir, "%s", argv[1]);
    grid = (int)atoi(argv[2]);
    start_num = (int)atoi(argv[3]);
    iter_num = (int)atoi(argv[4]);
    sigma = (double)atof(argv[5]);
    W_const = (double)atof(argv[6]);
    max = (double)atof(argv[7]);

    // threads to use
    int cores_used = (int)atoi(argv[8]);

    // omp_set_num_threads(cores_used);

    o_max = 1.0 / max;

    // load input parameters
    char l_dummy[200];
    double d_dummy;
    double d_dummy2;
    int i_dummy;

    FILE *l_fp;
    char l_fname[200];
    int count_scan;
    count_scan = 0;

    sprintf(l_fname, "%s/input.dat", work_dir);

    // read parameters, n_particles, thickness, and save steps
    l_fp = fopen(l_fname, "r");
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &N_particle);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &save_step);
    count_scan += fscanf(l_fp, "%s %lf %lf %lf", l_dummy, &d_dummy, &d_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &thickness);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);
    count_scan += fscanf(l_fp, "%s %lf", l_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %d", l_dummy, &i_dummy);

    // dev: add population parameters here
    count_scan += fscanf(l_fp, "%s %lf %lf", l_dummy, &d_dummy, &d_dummy);
    count_scan += fscanf(l_fp, "%s %lf %lf", l_dummy, &d_dummy, &d_dummy);

    fclose(l_fp);

    // save parameters to output_nematic_field.dat
    char l_output_file[200];
    char l_outstr[200];
    FILE *l_fpoutput;

    sprintf(l_output_file, "%s/output_nematic_field.dat", work_dir);
    l_fpoutput = fopen(l_output_file, "w");

    fprintf(l_fpoutput, "grid %d\n", grid);
    fprintf(l_fpoutput, "start %d\n", start_num);
    fprintf(l_fpoutput, "inter_num %d\n", iter_num);
    fprintf(l_fpoutput, "sigma %lf\n", sigma);
    fprintf(l_fpoutput, "W_const %lf\n", W_const);
    fprintf(l_fpoutput, "max %lf\n", max);

    fclose(l_fpoutput);

    // allocate memory for molecule array
    molecule = (struct molecule_var *)calloc(N_particle, sizeof(struct molecule_var));

    assert(molecule);

    // allocate memory for nematic field array
    nematic_field = (struct nematic_field_var *)calloc(grid * grid, sizeof(struct nematic_field_var));
    assert(nematic_field);

    int i, j, k;
    int ind;
    int time;
    double l_phi;
    double dx, dy, smooth;
    double l_active;
    int n_active;
    for (time = start_num; time < iter_num; time += save_step)
    {
        sprintf(l_fname, "%s/t-%d.pos", work_dir, time);
        l_fp = fopen(l_fname, "r");
        n_active = 0;
        for (i = 0; i < N_particle; i++)
        {

            count_scan = fread(&molecule[i].x, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].y, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].z, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].ex, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].ey, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].ez, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].length, sizeof(double), 1, l_fp);
            count_scan = fread(&l_active, sizeof(double), 1, l_fp);
            molecule[i].active = (int)l_active;
            count_scan = fread(&d_dummy, sizeof(double), 1, l_fp);
            count_scan = fread(&d_dummy, sizeof(double), 1, l_fp);
            count_scan = fread(&d_dummy, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].force_x, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].force_y, sizeof(double), 1, l_fp);
            count_scan = fread(&molecule[i].force_z, sizeof(double), 1, l_fp);

            int nm = 0;
            for (nm = 0; nm < 33; nm++)
                count_scan = fread(&d_dummy2, sizeof(double), 1, l_fp);

            l_phi = atan2(molecule[i].ey, molecule[i].ex);
            molecule[i].ex_p = -sin(l_phi);
            molecule[i].ey_p = cos(l_phi);
            i_dummy = (int)d_dummy;
            if (molecule[i].active == 1 && i_dummy > 0)
                n_active++;
        }
        fclose(l_fp);

        // // #pragma omp parallel for
        for (i = 0; i < grid; i++)
        {
            for (j = 0; j < grid; j++)
            {
                ind = i + j * grid;
                nematic_field[ind].rho = 0.0;
                nematic_field[ind].Q11 = 0.0;
                nematic_field[ind].Q12 = 0.0;
                nematic_field[ind].vx = 0.0;
                nematic_field[ind].vy = 0.0;
                nematic_field[ind].press = 0.0;
            }
        }

        // // #pragma omp parallel for
        for (i = 0; i < grid; i++)
        {
            for (j = 0; j < grid; j++)
            {
                ind = i + j * grid;
                nematic_field[ind].x = ((((double)i) - ((double)grid) / 2.0) + 0.5) / ((((double)grid) / 2.0) + 0.5) * max;
                nematic_field[ind].y = ((((double)j) - ((double)grid) / 2.0) + 0.5) / ((((double)grid) / 2.0) + 0.5) * max;
                for (k = 0; k < N_particle; k++)
                {
                    if (molecule[k].active == 1)
                    {
                        dx = (nematic_field[ind].x - molecule[k].x); //*o_max;
                        dy = (nematic_field[ind].y - molecule[k].y); //*o_max;
                        // dx=(dx-rint(dx))*max;
                        // dy=(dy-rint(dy))*max;

                        smooth = (0.5 * (tanh((molecule[k].ex * dx + molecule[k].ey * dy + 0.5 * molecule[k].length) / sigma) - tanh((molecule[k].ex * dx + molecule[k].ey * dy - 0.5 * molecule[k].length) / sigma)) * 0.5 * (tanh((molecule[k].ex_p * dx + molecule[k].ey_p * dy + 0.5 * W_const) / sigma) - tanh((molecule[k].ex_p * dx + molecule[k].ey_p * dy - 0.5 * W_const) / sigma)));
                        nematic_field[ind].rho += smooth;
                        nematic_field[ind].Q11 += smooth * (2.0 * molecule[k].ex * molecule[k].ex - 1.0);
                        nematic_field[ind].Q12 += smooth * (2.0 * molecule[k].ex * molecule[k].ey);
                        nematic_field[ind].vx += smooth * molecule[k].force_x;
                        nematic_field[ind].vy += smooth * molecule[k].force_y;
                        nematic_field[ind].press += smooth * sqrt(molecule[k].force_x * molecule[k].force_x + molecule[k].force_y * molecule[k].force_y) / (2.0 * molecule[k].length + 2.0 * thickness);
                    }
                }
            }
        }

        sprintf(l_fname, "%s/grid-%d.pos", work_dir, time);
        l_fp = fopen(l_fname, "w");
        for (i = 0; i < grid; i++)
        {
            for (j = 0; j < grid; j++)
            {
                ind = j + i * grid; // seems wrong but is needed to make it conform with numpy convention!
                fprintf(l_fp, "%.10f	", nematic_field[ind].x);
                fprintf(l_fp, "%.10f	", nematic_field[ind].y);
                fprintf(l_fp, "%.10f	", nematic_field[ind].rho);
                fprintf(l_fp, "%.10f	", nematic_field[ind].Q11);
                fprintf(l_fp, "%.10f	", nematic_field[ind].Q12);
                fprintf(l_fp, "%.10f	", nematic_field[ind].vx);
                fprintf(l_fp, "%.10f	", nematic_field[ind].vy);
                fprintf(l_fp, "%.10f	\n", nematic_field[ind].press);
            }
        }
        fclose(l_fp);
    }

    fflush(NULL);
    printf("All done\n");

    return 0;
}
