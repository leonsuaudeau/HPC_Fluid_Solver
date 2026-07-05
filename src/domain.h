#ifndef HPC_FLUID_SOLVER_DECOMPOSITION_H
#define HPC_FLUID_SOLVER_DECOMPOSITION_H

#include <Kokkos_Core.hpp>
#include "definitions.h"

#include <mpi.h>

namespace dom {

inline void exchange_halos_vert_strips(const Distribution_t &f,
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
        if (left != MPI_PROC_NULL) {
            f(0, y, i) = receive_buffer_left(index);
        }
        if (right != MPI_PROC_NULL) {
            f(rank_width + 1, y, i) = receive_buffer_right(index);
        }
    });
}

inline void exchange_halos_tiles_x_pass(const Distribution_t &f, const MPI_Comm &cart,
    const Vec &send_buffer_left, const Vec &send_buffer_right,
    const Vec &receive_buffer_left, const Vec &receive_buffer_right,
    const int left, const int right, const int tile_width, const int tile_height,
    const int halo_size[2], const Kokkos::Array<int, 3> &move_left_dirs, const Kokkos::Array<int, 3> &move_right_dirs) {

    // Put data in buffers
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {tile_height, 3}),
        KOKKOS_LAMBDA(const int y, const int i) {
        const int index = y * 3 + i;
        const int y_no_halo = y + 1;
        send_buffer_left(index) = f(1, y_no_halo, move_left_dirs[i]);
        send_buffer_right(index) = f(tile_width, y_no_halo, move_right_dirs[i]);
    });
    Kokkos::fence();

    // Send and receive
    MPI_Sendrecv(
        send_buffer_left.data(), halo_size[0], MPI_FLOAT, left, 0,
        receive_buffer_right.data(), halo_size[0], MPI_FLOAT, right, 0,
        cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        send_buffer_right.data(), halo_size[0], MPI_FLOAT, right, 1,
        receive_buffer_left.data(), halo_size[0], MPI_FLOAT, left, 1,
        cart, MPI_STATUS_IGNORE);

    // Put data in distribution
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {tile_height, 3}),
        KOKKOS_LAMBDA(const int y, const int i) {
        const int index = y * 3 + i;
        const int y_no_halo = y + 1;

        // important! we need to flip up/down here because we are now looking
        // at the incoming, not the outgoing channels
        if (left != MPI_PROC_NULL) {
            f(0, y_no_halo, move_right_dirs[i]) = receive_buffer_left(index);
        }
        if (right != MPI_PROC_NULL) {
            f(tile_width + 1, y_no_halo, move_left_dirs[i]) = receive_buffer_right(index);
        }
    });
}

inline void exchange_halos_tiles_y_pass(const Distribution_t &f, const MPI_Comm &cart,
    const Vec &send_buffer_up, const Vec &send_buffer_down,
    const Vec &receive_buffer_up, const Vec &receive_buffer_down,
    const int up, const int down, const int tile_width, const int tile_height,
    const int halo_size[2], const Kokkos::Array<int, 3> &move_up_dirs, const Kokkos::Array<int, 3> &move_down_dirs) {

    // Put data in buffers
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {tile_width + 2, 3}),
        KOKKOS_LAMBDA(const int x, const int i) {
        const int index = x * 3 + i;
        send_buffer_up(index) = f(x, 1, move_up_dirs[i]);
        send_buffer_down(index) = f(x, tile_height, move_down_dirs[i]);
    });
    Kokkos::fence();

    // Send and receive
    MPI_Sendrecv(
        send_buffer_down.data(), halo_size[1], MPI_FLOAT, down, 1,
        receive_buffer_up.data(), halo_size[1], MPI_FLOAT, up, 1,
        cart, MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        send_buffer_up.data(), halo_size[1], MPI_FLOAT, up, 0,
        receive_buffer_down.data(), halo_size[1], MPI_FLOAT, down, 0,
        cart, MPI_STATUS_IGNORE);

    // Put data in distribution
    Kokkos::parallel_for("send buffer copy",
        Kokkos::MDRangePolicy({0, 0}, {tile_width + 2, 3}),
        KOKKOS_LAMBDA(const int x, const int i) {
        const int index = x * 3 + i;
        if (down != MPI_PROC_NULL) {
            f(x, 0, move_up_dirs[i]) = receive_buffer_down(index);
        }
        if (up != MPI_PROC_NULL) {
            f(x, tile_height + 1, move_down_dirs[i]) = receive_buffer_up(index);
        }
    });
}

}

#endif
