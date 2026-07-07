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
    const iVec &c_x, const iVec &c_y, const Kokkos::Array<int, 2> &boundary_conditions,
    const Kokkos::Array<float, 4> &boundary_values, const iVec &opposite_i,
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
void streaming_with_boundaries_mpi_strips(
    const Distribution_t &f, const Distribution_t &f_new, const Vec &w,
    const iVec &c_x, const iVec &c_y, const Kokkos::Array<int, 2> &boundary_conditions,
    const Kokkos::Array<float, 4> &boundary_values, const iVec &opposite_i,
    const int local_x, const int x_offset, const int y, const int grid_width, const int grid_height) {

    const int global_x = local_x + x_offset - 1;

    for (int i = 0; i < 9; i++) {
        const int rank_x_source = local_x - c_x(i);
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

            f_new(local_x, y, i) = f(local_x, y, opposite_i(i)) + calculate_correction_term(
                w(i), 1.0f, c_x(i), c_y(i),
                v_x_wall , v_y_wall);
        } else {
            // periodic or no boundary
            if (global_y_source < 0 || global_y_source >= grid_height) {
                global_y_source = wrap(global_y_source, grid_height);
            }

            f_new(local_x, y, i) = f(rank_x_source, global_y_source, i);
        }
    }
}

KOKKOS_INLINE_FUNCTION
float relax_single_tile(const float f_i, const float rho,
    const float u_x, const float u_y, const int c_x_i, const int c_y_i,
    const float omega, const float w_i, const float norm_u_squared) {
    const float c_i_dot_u = c_x_i * u_x + c_y_i * u_y;
    return f_i + omega * (w_i * rho * (1 + 3 * c_i_dot_u + 4.5f * c_i_dot_u * c_i_dot_u - 1.5f * norm_u_squared) - f_i);
}

KOKKOS_INLINE_FUNCTION
void main_kernel_interior(const Distribution_t &f, const Distribution_t &f_new,
    const int local_x, const int local_y, const float omega) {
    // pull
    const float f_0 = f(local_x, local_y, 0);
    const float f_1 = f(local_x - 1, local_y, 1);
    const float f_2 = f(local_x, local_y - 1, 2);
    const float f_3 = f(local_x + 1, local_y, 3);
    const float f_4 = f(local_x, local_y + 1, 4);
    const float f_5 = f(local_x - 1, local_y - 1, 5);
    const float f_6 = f(local_x + 1, local_y - 1, 6);
    const float f_7 = f(local_x + 1, local_y + 1, 7);
    const float f_8 = f(local_x - 1, local_y + 1, 8);

    // calculate rho and velocity locally
    const float rho = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
    const float u_x = (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) / rho;
    const float u_y = (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) / rho;

    // relaxation
    const float norm_u_squared = u_x * u_x + u_y * u_y;

    f_new(local_x, local_y, 0) = relax_single_tile(f_0, rho, u_x, u_y, 0, 0, omega, 4.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 1) = relax_single_tile(f_1, rho, u_x, u_y, 1, 0, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 2) = relax_single_tile(f_2, rho, u_x, u_y, 0, 1, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 3) = relax_single_tile(f_3, rho, u_x, u_y, -1, 0, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 4) = relax_single_tile(f_4, rho, u_x, u_y, 0, -1, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 5) = relax_single_tile(f_5, rho, u_x, u_y, 1, 1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 6) = relax_single_tile(f_6, rho, u_x, u_y, -1, 1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 7) = relax_single_tile(f_7, rho, u_x, u_y, -1, -1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 8) = relax_single_tile(f_8, rho, u_x, u_y, 1, -1, omega, 1.0f / 36.0f, norm_u_squared);
}

KOKKOS_INLINE_FUNCTION
float pull_or_collide_single_tile(const Distribution_t &f,
    const Kokkos::Array<int, 2> &boundary_conditions,
    const Kokkos::Array<float, 4> &boundary_values,
    const int local_x, const int local_y, const int global_x, const int global_y,
    const int grid_width, const int grid_height, const int i, const int c_x_i, const int c_y_i,
    const int opposite_i, const float w_i) {

    const int rank_x_source = local_x - c_x_i;
    const int rank_y_source = local_y - c_y_i;
    const int global_x_source = global_x - c_x_i;
    const int global_y_source = global_y - c_y_i;

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

        return f(local_x, local_y, opposite_i)
            + calculate_correction_term(w_i, 1.0f,
                c_x_i, c_y_i, v_x_wall , v_y_wall);
    }

    return f(rank_x_source, rank_y_source, i);
}



KOKKOS_INLINE_FUNCTION
void main_kernel_boundary(const Distribution_t &f, const Distribution_t &f_new,
    const Kokkos::Array<int, 2> &boundary_conditions,
    const Kokkos::Array<float, 4> &boundary_values,
    const int local_x, const int local_y, const int x_offset, const int y_offset,
    const int grid_width, const int grid_height, const float omega) {

    // pull or collide
    const int global_x = local_x + x_offset - 1;
    const int global_y = local_y + y_offset - 1;

    const float f_0 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 0, 0, 0, 0, 4.0f / 9.0f);
    const float f_1 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 1, 1, 0, 3, 1.0f / 9.0f);
    const float f_2 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 2, 0, 1, 4, 1.0f / 9.0f);
    const float f_3 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 3, -1, 0, 1, 1.0f / 9.0f);
    const float f_4 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 4, 0, -1, 2, 1.0f / 9.0f);
    const float f_5 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 5, 1, 1, 7, 1.0f / 36.0f);
    const float f_6 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 6, -1, 1, 8, 1.0f / 36.0f);
    const float f_7 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 7, -1, -1, 5, 1.0f / 36.0f);
    const float f_8 = pull_or_collide_single_tile(f, boundary_conditions, boundary_values, local_x, local_y, global_x, global_y, grid_width, grid_height, 8, 1, -1, 6, 1.0f / 36.0f);

    // calculate rho and velocity locally
    const float rho = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
    const float u_x = (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) / rho;
    const float u_y = (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) / rho;

    // relaxation
    const float norm_u_squared = u_x * u_x + u_y * u_y;

    f_new(local_x, local_y, 0) = relax_single_tile(f_0, rho, u_x, u_y, 0, 0, omega, 4.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 1) = relax_single_tile(f_1, rho, u_x, u_y, 1, 0, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 2) = relax_single_tile(f_2, rho, u_x, u_y, 0, 1, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 3) = relax_single_tile(f_3, rho, u_x, u_y, -1, 0, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 4) = relax_single_tile(f_4, rho, u_x, u_y, 0, -1, omega, 1.0f / 9.0f, norm_u_squared);
    f_new(local_x, local_y, 5) = relax_single_tile(f_5, rho, u_x, u_y, 1, 1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 6) = relax_single_tile(f_6, rho, u_x, u_y, -1, 1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 7) = relax_single_tile(f_7, rho, u_x, u_y, -1, -1, omega, 1.0f / 36.0f, norm_u_squared);
    f_new(local_x, local_y, 8) = relax_single_tile(f_8, rho, u_x, u_y, 1, -1, omega, 1.0f / 36.0f, norm_u_squared);
}

KOKKOS_INLINE_FUNCTION
void streaming_with_boundaries_mpi_tiles(
    const Distribution_t &f, const Distribution_t &f_new, const Vec &w,
    const iVec &c_x, const iVec &c_y, const Kokkos::Array<int, 2> &boundary_conditions,
    const Kokkos::Array<float, 4> &boundary_values, const iVec &opposite_i,
    const int local_x, const int local_y, const int x_offset, const int y_offset,
    const int grid_width, const int grid_height) {

    const int global_x = local_x + x_offset - 1;
    const int global_y = local_y + y_offset - 1;

    for (int i = 0; i < 9; i++) {
        const int rank_x_source = local_x - c_x(i);
        const int global_x_source = global_x - c_x(i);
        const int rank_y_source = local_y - c_y(i);
        const int global_y_source = global_y - c_y(i);

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

            f_new(local_x, local_y, i) = f(local_x, local_y, opposite_i(i))
                + calculate_correction_term(w(i), 1.0f,
                    c_x(i), c_y(i), v_x_wall , v_y_wall);
        } else {
            f_new(local_x, local_y, i) = f(rank_x_source, rank_y_source, i);
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

inline float calculate_reynolds_number(const Kokkos::Array<float, 4> &lid_vel, const int L, const float omega) {
    const float nu = 1.0f/3.0f * (1.0f/omega - 1.0f/2.0f);
    return std::max(std::max(lid_vel[0], lid_vel[1]), std::max(lid_vel[2], lid_vel[3])) * static_cast<float>(L) / nu;
}

inline double blups(const int width, const int height, const int max_steps, const double walltime) {
    const std::int64_t lattice_updates =
        static_cast<std::int64_t>(width) *
        static_cast<std::int64_t>(height) *
        static_cast<std::int64_t>(max_steps);

    return static_cast<double>(lattice_updates) / walltime / 1.0e9;
}

}

#endif // HPC_FLUID_SOLVER_EQUATIONS_H
