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
float calculate_correction_term(
    const float w_i, const  float rho_wall,
    const int c_x_i, const int c_y_i,
    const float v_x_wall, const float v_y_wall) {
    const float c_i_dot_u = c_x_i * v_x_wall + c_y_i * v_y_wall;
    return 6.0f * w_i * rho_wall * c_i_dot_u;
}

KOKKOS_INLINE_FUNCTION
void streaming_with_boundaries(
    const Distribution_t &f, const Distribution_t &f_new, const Vec &w,
    const iVec &c_x, const iVec &c_y, const int boundary_conditions[2],
    const float boundary_values[4], const iVec &opposite_i,
    const int x, const int y, const int grid_width, const int grid_height) {
    for (int i = 0; i < 9; i++) {
        int x_source = x - c_x(i);
        int y_source = y - c_y(i);

        int boundary = -1;

        if (x_source < 0) {
            boundary = 0;
        } else if (y_source < 0) {
            boundary = 1;
        } else if (x_source >= grid_width) {
            boundary = 2;
        } else if (y_source >= grid_height) {
            boundary = 3;
        }

        // no boundary
        const int x_periodic = wrap(x_source, grid_width);
        const int y_periodic = wrap(y_source, grid_height);
        f_new(x, y, i) = f(x_periodic, y_periodic, i);

        if (boundary >= 0 && boundary_conditions[boundary % 2] == 1) {
            const float wall_speed = boundary_values[boundary];

            float v_x_wall = 0.0f;
            float v_y_wall = 0.0f;

            if (boundary == 1 || boundary == 3) {
                v_x_wall = wall_speed;
            }else {
                v_y_wall = wall_speed;
            }

            f_new(x, y, i) = f(x, y, opposite_i(i)) + calculate_correction_term(
                w(i), 1.0f, c_x(i), c_y(i),
                v_x_wall , v_y_wall);
        }
    }
}

KOKKOS_INLINE_FUNCTION
void streaming_with_boundaries_mpi(
    const Distribution_t &f, const Distribution_t &f_new, const Vec &w,
    const iVec &c_x, const iVec &c_y, const int boundary_conditions[2],
    const float boundary_values[4], const iVec &opposite_i,
    const int rank_x, const int x_offset, const int y, const int grid_width, const int grid_height) {

    const int global_x = rank_x + x_offset - 1;

    for (int i = 0; i < 9; i++) {
        const int rank_x_source = rank_x - c_x(i);
        const int global_x_source = global_x - c_x(i);
        int global_y_source = y - c_y(i);

        int boundary = -1;

        if (global_x_source < 0) {
            boundary = 0;
        } else if (global_y_source < 0) {
            boundary = 1;
        } else if (global_x_source >= grid_width) {
            boundary = 2;
        } else if (global_y_source >= grid_height) {
            boundary = 3;
        }

        if (boundary >= 0 && boundary_conditions[boundary % 2] == 1) {
            // wall boundary
            const float wall_speed = boundary_values[boundary];

            float v_x_wall = 0.0f;
            float v_y_wall = 0.0f;

            if (boundary == 1 || boundary == 3) {
                v_x_wall = wall_speed;
            }else {
                v_y_wall = wall_speed;
            }

            f_new(rank_x, y, i) = f(rank_x, y, opposite_i(i)) + calculate_correction_term(
                w(i), 1.0f, c_x(i), c_y(i),
                v_x_wall , v_y_wall);
        } else {
            // periodic or no boundary
            if (global_y_source < 0 || global_y_source >= grid_height) {
                global_y_source = wrap(global_y_source, grid_height);
            }

            f_new(rank_x, y, i) = f(rank_x_source, global_y_source, i);
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
