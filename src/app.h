#ifndef HPC_FLUID_SOLVER_APP_H
#define HPC_FLUID_SOLVER_APP_H
#include "app_state.h"
#include "grid.h"

class App {
public:
    App();
    int run(int argc, char *argv[]);
    AppState state{};
    Grid grid;
};

#endif // HPC_FLUID_SOLVER_APP_H
