#ifndef HPC_FLUID_SOLVER_EQUATIONS_H
#define HPC_FLUID_SOLVER_EQUATIONS_H

#include <Kokkos_Core.hpp>
#include "definitions.h"

namespace eq {
KOKKOS_INLINE_FUNCTION
int wrap(const int i, const int m) {
    return (i % m + m) % m;
}

KOKKOS_INLINE_FUNCTION
void update_density(
    const Density_t &rho, const Distribution_t &f,
    const int x, const int y) {
    float sum = 0.0f;
    for (int i = 0; i < 9; i++) {
        sum += f(x, y, i);
    }
    rho(x, y) = sum;
}

KOKKOS_INLINE_FUNCTION
void update_velocity(
    const Velocity_t &v_x, const Velocity_t &v_y,
    const iVec &c_x, const iVec &c_y, const Density_t &rho,
    const Distribution_t &f, const int x, const int y) {
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (int i = 0; i < 9; i++) {
        sum_x += f(x, y, i) * c_x(i);
        sum_y += f(x, y, i) * c_y(i);
    }
    v_x(x, y) = sum_x / rho(x, y);
    v_y(x, y) = sum_y / rho(x, y);
}

KOKKOS_INLINE_FUNCTION
void streaming(
    const Distribution_t &f, const Distribution_t &f_new,
    const iVec &c_x, const iVec &c_y, const int x, const int y,
    const int grid_width, const int grid_height) {
    for (int i = 0; i < 9; i++) {
        const int x_old = wrap(x - c_x(i), grid_width);
        const int y_old = wrap(y - c_y(i), grid_height);
        f_new(x, y, i) = f(x_old, y_old, i);
    }
}

KOKKOS_INLINE_FUNCTION
void streaming_with_boundaries(
    const Distribution_t &f, const Distribution_t &f_new,
    const iVec &c_x, const iVec &c_y, const Grid_Mask &sf_mask, const iVec &opposite_i,
    const int x, const int y, const int grid_width, const int grid_height) {
    for (int i = 0; i < 9; i++) {
        const int x_old = wrap(x - c_x(i), grid_width);
        const int y_old = wrap(y - c_y(i), grid_height);

        if (sf_mask(x_old, y_old) == 1) {
            f_new(x, y, i) = f(x, y, opposite_i(i));
        }else {
            f_new(x, y, i) = f(x_old, y_old, i);
        }
    }
}

KOKKOS_INLINE_FUNCTION
float calculate_eq_distrib(
    const float rho_cell, const float v_x_cell,
    const float v_y_cell, const int c_x_i,
    const int c_y_i, const float w_i) {
    const float c_i_dot_u = c_x_i * v_x_cell + c_y_i * v_y_cell;
    const float norm_u_squared = v_x_cell * v_x_cell + v_y_cell * v_y_cell;
    return w_i * rho_cell * (1 + 3 * c_i_dot_u + 4.5f * c_i_dot_u * c_i_dot_u - 1.5f * norm_u_squared);
}

KOKKOS_INLINE_FUNCTION
void relaxation(
    const Distribution_t &f, const Density_t &rho,
    const Velocity_t &v_x, const Velocity_t &v_y,
    const iVec &c_x, const iVec &c_y, const Vec &w,
    const float omega, const int x, const int y) {
    for (int i = 0; i < 9; i++) {
        f(x, y, i) += omega * (calculate_eq_distrib(rho(x, y), v_x(x, y), v_y(x, y), c_x(i), c_y(i), w(i)) - f(x, y, i));
    }
}

inline float sinusoidal(const int x, const float epsilon, const float k) {
    return epsilon * sinf(k * static_cast<float>(x));
}

KOKKOS_INLINE_FUNCTION
float calculate_sin_amplitude(const Velocity_t &v_x, const Velocity_t &v_y, const float k) {
    const int n_x = static_cast<int>(v_x.extent(0));
    const int n_y = static_cast<int>(v_y.extent(1));
    const float n_x_inv = 1.0f / static_cast<float>(n_x);
    const float n_y_inv = 1.0f / static_cast<float>(n_y);

    float sum = 0.0f;

    for (int x = 0; x < n_x; x++) {
        for (int y = 0; y < n_y; y++) {
            sum += v_x(x, y) * sinf(k * static_cast<float>(y));
        }
    }
    return sum * n_x_inv * n_y_inv * 2.0f;
}
}

#endif // HPC_FLUID_SOLVER_EQUATIONS_H