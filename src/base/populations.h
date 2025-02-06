#ifndef POPULATIOS
#define POPULATIOS

void grow();
void init_cells();
void starting_cells();
int annulus(int start_idx, bool active, int label, bool grown);
int annulus(int start_idx, bool active = true, int label = 1, bool grown = false);
int random_cells(int start_idx, int Start_parts, int _label);
int tracer_cells(int start_idx);

#endif