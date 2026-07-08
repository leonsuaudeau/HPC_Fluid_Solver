#ifndef HPC_FLUID_SOLVER_DEFINITIONS_H
#define HPC_FLUID_SOLVER_DEFINITIONS_H

using Distribution_t = Kokkos::View<float***, Kokkos::LayoutLeft> ;
using Distribution_t_flat = Kokkos::View<float*, Kokkos::LayoutLeft> ;
using Velocity_t = Kokkos::View<float**>;
using Density_t = Kokkos::View<float**>;
using Grid_Mask = Kokkos::View<int**>;
using iVec = Kokkos::View<int*>;
using Vec = Kokkos::View<float*>;
using iVec_host = Kokkos::View<int*, Kokkos::HostSpace>;
using Vec_host = Kokkos::View<float*, Kokkos::HostSpace>;
using Left_Policy = Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>;

#endif // HPC_FLUID_SOLVER_DEFINITIONS_H
