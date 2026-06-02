#ifndef HPC_FLUID_SOLVER_APP_STATE_H
#define HPC_FLUID_SOLVER_APP_STATE_H

struct AppState {
    int timestep = 0;
    int max_steps = 100;
    int draw_steps = 1;
};

#endif // HPC_FLUID_SOLVER_APP_STATE_H
