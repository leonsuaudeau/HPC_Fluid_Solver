#ifndef HPC_FLUID_SOLVER_APP_H
#define HPC_FLUID_SOLVER_APP_H
#include "app_state.h"

class App {
public:
    App();
    int run(int argc, char *argv[]);
    int run_mpi(int argc, char *argv[]) const;
    AppState state{};
};

#endif // HPC_FLUID_SOLVER_APP_H
