#ifndef HPC_FLUID_SOLVER_DEFINITIONS_H
#define HPC_FLUID_SOLVER_DEFINITIONS_H

using Distribution_t = Kokkos::View<float***> ;
using Velocity_t = Kokkos::View<float**>;
using Density_t = Kokkos::View<float**>;
using Grid_Mask = Kokkos::View<int**>;
using iVec = Kokkos::View<int*>;
using Vec = Kokkos::View<float*>;
using iVec_host = Kokkos::View<int*, Kokkos::HostSpace>;
using Vec_host = Kokkos::View<float*, Kokkos::HostSpace>;

#endif // HPC_FLUID_SOLVER_DEFINITIONS_H
