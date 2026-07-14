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

inline void exchange_halos_tiles_x_pass(
    const Distribution_t_flat &f, const MPI_Comm &cart,
    const Vec &send_buffer_left, const Vec &send_buffer_right,
    const Vec &receive_buffer_left, const Vec &receive_buffer_right,
    const int left, const int right, const int tile_width,
    const int tile_height, const int halo_size[]) {

    const int width_with_halos = tile_width + 2;
    const int offset = width_with_halos * (tile_height + 2);

    // Put data in buffers
    Kokkos::parallel_for("send buffer copy",
        Kokkos::RangePolicy(0, tile_height),
        KOKKOS_LAMBDA(const int y) {
        const int width_y = width_with_halos * (y + 1);
        const int base_left = 1 + width_y;
        const int base_right = tile_width + width_y;

        const int y_3 = y * 3;
        const int buf_idx_1 = y_3 + 1;
        const int buf_idx_2 = y_3 + 2;

        send_buffer_left(y_3) = f(base_left + 3 * offset);
        send_buffer_right(y_3) = f(base_right + 1 * offset);

        send_buffer_left(buf_idx_1) = f(base_left + 6 * offset);
        send_buffer_right(buf_idx_1) = f(base_right + 5 * offset);

        send_buffer_left(buf_idx_2) = f(base_left + 7 * offset);
        send_buffer_right(buf_idx_2) = f(base_right + 8 * offset);
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
        Kokkos::RangePolicy(0, tile_height),
        KOKKOS_LAMBDA(const int y) {
        const int width_y = width_with_halos * (y + 1);
        const int base_left = 0 + width_y;
        const int base_right = tile_width + 1 + width_y;

        const int y_3 = y * 3;
        const int buf_idx_1 = y_3 + 1;
        const int buf_idx_2 = y_3 + 2;

        // important! we need to flip left/right here because we are now looking
        // at the incoming, not the outgoing channels
        if (left != MPI_PROC_NULL) {
            f(base_left + 1 * offset) = receive_buffer_left(y_3);
            f(base_left + 5 * offset) = receive_buffer_left(buf_idx_1);
            f(base_left + 8 * offset) = receive_buffer_left(buf_idx_2);
        }
        if (right != MPI_PROC_NULL) {
            f(base_right + 3 * offset) = receive_buffer_right(y_3);
            f(base_right + 6 * offset) = receive_buffer_right(buf_idx_1);
            f(base_right + 7 * offset) = receive_buffer_right(buf_idx_2);
        }
    });
}

inline void exchange_halos_tiles_y_pass(
    const Distribution_t_flat &f, const MPI_Comm &cart,
    const Vec &send_buffer_up, const Vec &send_buffer_down,
    const Vec &receive_buffer_up, const Vec &receive_buffer_down, const int up,
    const int down, const int tile_width, const int tile_height,
    const int halo_size[]) {

    const int width_with_halos = tile_width + 2;
    const int offset = width_with_halos * (tile_height + 2);

    // Put data in buffers
    Kokkos::parallel_for("send buffer copy",
        Kokkos::RangePolicy(0, tile_width + 2),
        KOKKOS_LAMBDA(const int x) {
        const int base_down = x + width_with_halos * 1;
        const int base_up = x + width_with_halos * tile_height;

        const int x_3 = x * 3;
        const int buf_idx_1 = x_3 + 1;
        const int buf_idx_2 = x_3 + 2;

        send_buffer_down(x_3) = f(base_down + 4 * offset);
        send_buffer_up(x_3) = f(base_up + 2 * offset);

        send_buffer_down(buf_idx_1) = f(base_down + 7 * offset);
        send_buffer_up(buf_idx_1) = f(base_up + 5 * offset);

        send_buffer_down(buf_idx_2) = f(base_down + 8 * offset);
        send_buffer_up(buf_idx_2) = f(base_up + 6 * offset);
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
        Kokkos::RangePolicy(0, tile_width + 2),
        KOKKOS_LAMBDA(const int x) {
        // base_down = x + width_with_halos * 0 = x;
        const int base_up = x + width_with_halos * (tile_height + 1);

        const int x_3 = x * 3;
        const int buf_idx_1 = x_3 + 1;
        const int buf_idx_2 = x_3 + 2;

        if (down != MPI_PROC_NULL) {
            f(x + 2 * offset) = receive_buffer_down(x_3);
            f(x + 5 * offset) = receive_buffer_down(buf_idx_1);
            f(x + 6 * offset) = receive_buffer_down(buf_idx_2);
        }
        if (up != MPI_PROC_NULL) {
            f(base_up + 4 * offset) = receive_buffer_up(x_3);
            f(base_up + 7 * offset) = receive_buffer_up(buf_idx_1);
            f(base_up + 8 * offset) = receive_buffer_up(buf_idx_2);
        }
    });
}

}

#endif
