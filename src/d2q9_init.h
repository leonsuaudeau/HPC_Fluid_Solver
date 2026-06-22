#ifndef HPC_FLUID_SOLVER_D2Q9_INIT_H
#define HPC_FLUID_SOLVER_D2Q9_INIT_H

#include <Kokkos_Core.hpp>
#include "definitions.h"

namespace d2q9 {

iVec_host c_x_host_init(const iVec& c_x_device) {
    const auto c_x_host = Kokkos::create_mirror_view(c_x_device);
    c_x_host(0) = 0;
    c_x_host(1) = 1;
    c_x_host(2) = 0;
    c_x_host(3) = -1;
    c_x_host(4) = 0;
    c_x_host(5) = 1;
    c_x_host(6) = -1;
    c_x_host(7) = -1;
    c_x_host(8) = 1;
    Kokkos::deep_copy(c_x_device, c_x_host);
    return c_x_host;
}

iVec_host c_y_host_init(const iVec &c_y_device) {
    const auto c_y_host = Kokkos::create_mirror_view(c_y_device);
    c_y_host(0) = 0;
    c_y_host(1) = 0;
    c_y_host(2) = 1;
    c_y_host(3) = 0;
    c_y_host(4) = -1;
    c_y_host(5) = 1;
    c_y_host(6) = 1;
    c_y_host(7) = -1;
    c_y_host(8) = -1;
    Kokkos::deep_copy(c_y_device, c_y_host);
    return c_y_host;
}

Vec_host w_host_init(const Vec &w_device) {
    auto w_host = Kokkos::create_mirror_view(w_device);
    w_host(0) = 4.0f / 9.0f;
    w_host(1) = 1.0f / 9.0f;
    w_host(2) = 1.0f / 9.0f;
    w_host(3) = 1.0f / 9.0f;
    w_host(4) = 1.0f / 9.0f;
    w_host(5) = 1.0f / 36.0f;
    w_host(6) = 1.0f / 36.0f;
    w_host(7) = 1.0f / 36.0f;
    w_host(8) = 1.0f / 36.0f;
    Kokkos::deep_copy(w_device, w_host);
    return w_host;
}

iVec_host opposite_i_host_init(const iVec &opposite_i_device) {
    auto opposite_i_host = Kokkos::create_mirror_view(opposite_i_device);
    opposite_i_host[0] = 0;
    opposite_i_host[1] = 3;
    opposite_i_host[2] = 4;
    opposite_i_host[3] = 1;
    opposite_i_host[4] = 2;
    opposite_i_host[5] = 7;
    opposite_i_host[6] = 8;
    opposite_i_host[7] = 5;
    opposite_i_host[8] = 6;
    Kokkos::deep_copy(opposite_i_device, opposite_i_host);
    return opposite_i_host;
}
}

#endif // HPC_FLUID_SOLVER_D2Q9_INIT_H

