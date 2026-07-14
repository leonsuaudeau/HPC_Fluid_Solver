#ifndef HPC_FLUID_SOLVER_APP_H
#define HPC_FLUID_SOLVER_APP_H
#include "app_state.h"

enum output_mode {
    moving_lid_single,
    moving_lid_animation,
    shear_wave_over_time,
    shear_wave_different_k,
    validation,
};

class App {
public:
    App();
    int run(int argc, char *argv[]);
    int run_mpi_strips(int argc, char *argv[]) const;
    int run_mpi_tiles(int argc, char *argv[]) const;
    AppState state{};
};

#endif // HPC_FLUID_SOLVER_APP_H
