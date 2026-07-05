#include "app.h"

int main(int argc, char *argv[]) {
    App app;
    //return app.run(argc, argv);
    //return app.run_mpi_strips(argc, argv);
    return app.run_mpi_tiles(argc, argv);
}
