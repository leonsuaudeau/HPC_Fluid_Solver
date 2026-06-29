#ifndef HPC_FLUID_SOLVER_DECOMPOSITION_H
#define HPC_FLUID_SOLVER_DECOMPOSITION_H

#include <Kokkos_Core.hpp>
#include "definitions.h"

#include <mpi.h>

namespace dom {

inline void exchange_halos(const Distribution_t &f,
    const Vec &send_buffer_left, const Vec &send_buffer_right,
    const Vec &receive_buffer_left, const Vec &receive_buffer_right,
    const int left, const int right, const int rank_width, const int height, const int halo_size) {

    // Put data in buffers
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {height, 9}),
        KOKKOS_LAMBDA(const int y, const int i) {
        const int index = y * 9 + i;
        send_buffer_left(index) = f(1, y, i);
        send_buffer_right(index) = f(rank_width, y, i);
    });
    Kokkos::fence();

    // Send and receive
    MPI_Sendrecv(
        send_buffer_left.data(), halo_size, MPI_FLOAT, left, 0,
        receive_buffer_right.data(), halo_size, MPI_FLOAT, right, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        send_buffer_right.data(), halo_size, MPI_FLOAT, right, 1,
        receive_buffer_left.data(), halo_size, MPI_FLOAT, left, 1,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Put data in distribution
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {height, 9}),
        KOKKOS_LAMBDA(const int y, const int i) {
        const int index = y * 9 + i;
        f(0, y, i) = receive_buffer_left(index);
        f(rank_width + 1, y, i) = receive_buffer_right(index);
    });
}

}

#endif