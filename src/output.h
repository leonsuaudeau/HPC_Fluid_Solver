#ifndef HPC_FLUID_SOLVER_OUTPUT_H
#define HPC_FLUID_SOLVER_OUTPUT_H
#include <string>
#include <Kokkos_Core.hpp>

namespace out {
void write_to(const std::string& name, const Kokkos::View<float**>& data, int width, int height, int frame);
void write_to(const std::string& name, const Kokkos::View<float*>& data);
}

#endif // HPC_FLUID_SOLVER_OUTPUT_H