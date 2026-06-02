#ifndef HPC_FLUID_SOLVER_OUTPUT_H
#define HPC_FLUID_SOLVER_OUTPUT_H
#include <string>
#include <Kokkos_Core.hpp>

namespace out {
void write_to(std::string name, Kokkos::View<float**> data, int width, int height, int frame);
}

#endif // HPC_FLUID_SOLVER_OUTPUT_H