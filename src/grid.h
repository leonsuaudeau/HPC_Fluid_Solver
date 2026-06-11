#ifndef HPC_FLUID_SOLVER_GRID_H
#define HPC_FLUID_SOLVER_GRID_H
#include <vector>

struct Grid {
    Grid(int width, int height);
    int width, height;
};

inline Grid::Grid(const int width, const int height) {
    this->width = width;
    this->height = height;
}

#endif // HPC_FLUID_SOLVER_GRID_H
