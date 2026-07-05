#ifndef HPC_FLUID_SOLVER_APP_STATE_H
#define HPC_FLUID_SOLVER_APP_STATE_H

struct AppState {
    int timestep = 0;
    int max_steps = 10000;
    int measurement_interval = 1000;
    int draw_steps = 2000;
};

#endif // HPC_FLUID_SOLVER_APP_STATE_H
