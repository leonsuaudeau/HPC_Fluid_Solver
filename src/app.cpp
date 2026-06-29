#include "app.h"
#include "d2q9_init.h"
#include "equations.h"
#include "output.h"
#include "domain.h"
#include <Kokkos_Core.hpp>
#include <iomanip>
#include <iostream>
#include <mpi.h>

App::App() = default;

int App::run(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        int width = 1000;
        int height = 1000;

        iVec c_x("c_x", 9);
        iVec c_y("c_y", 9);
        Vec w("w", 9);
        iVec opposite_i("opposite_i", 9);
        auto c_x_host = d2q9::c_x_host_init(c_x);
        auto c_y_host = d2q9::c_y_host_init(c_y);
        auto w_host = d2q9::w_host_init(w);
        auto opposite_i_host = d2q9::opposite_i_host_init(opposite_i);

        Density_t rho("rho", width, height);
        Velocity_t v_x("v_x", width, height);
        Velocity_t v_y("v_dty", width, height);
        Distribution_t f("f", width, height, 9);
        Distribution_t f_new("f_new", width, height, 9);

        int boundary_conditions [2] = {1,1}; // (horizontal, vertical) 0 = periodic, 1 = wall
        float boundary_values [4] = {0.0f, 0.0f, 0.0f, 0.1f}; // speed of wall, ignored if it isn't a wall
        // TODO: fix corners, use 2 vel. values per wall for this!

        auto rho_host = Kokkos::create_mirror_view(rho);
        auto v_x_host = Kokkos::create_mirror_view(v_x);
        auto v_y_host = Kokkos::create_mirror_view(v_y);
        auto f_host = Kokkos::create_mirror_view(f);

        Vec_host measurements_host("m", state.max_steps);

        // Grid initializations
        float n_y_inv = 1.0f / static_cast<float>(height);
        float k = 2.0f * 3.14159265359f * n_y_inv;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                rho_host(x, y) = 1.0f;
                v_x_host(x, y) = 0.0f;
                v_y_host(x, y) = 0.0f;

                for (int i = 0; i < 9; i++) {
                    f_host(x, y, i) = eq::calculate_eq_distrib(
                        rho_host(x, y),
                        v_x_host(x, y), v_y_host(x, y),
                        c_x_host(i), c_y_host(i),
                        w_host(i));
                }
            }
        }

        Kokkos::deep_copy(rho, rho_host);
        Kokkos::deep_copy(v_x, v_x_host);
        Kokkos::deep_copy(v_y, v_y_host);
        Kokkos::deep_copy(f, f_host);

        float global_mass = 0.0f;

        while (state.timestep < state.max_steps) {
            // Simulation step
            Kokkos::parallel_for("density + velocity + relaxation step",
                Kokkos::MDRangePolicy({0,0}, {width, height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                eq::update_density(rho, f, x, y);
                eq::update_velocity(v_x, v_y, c_x, c_y, rho, f, x, y);
            });

            if (state.timestep % state.measurement_interval == 0) {
                Kokkos::parallel_reduce("sum of masses",
                    Kokkos::MDRangePolicy({0,0}, {width, height}),
                    KOKKOS_LAMBDA(const int x, const int y, float& sum) {
                    sum += rho(x, y);
                },
                global_mass);
                std::cout << "Timestep " << state.timestep << "/" << state.max_steps << " | Mass: " << global_mass << std::endl;
            }

            Kokkos::parallel_for("density + velocity + relaxation step",
                Kokkos::MDRangePolicy({0,0}, {width, height}),
                KOKKOS_LAMBDA(const int x, const int y) {

            });

            Kokkos::parallel_for("streaming step",
                Kokkos::MDRangePolicy({0,0}, {width, height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                eq::streaming_with_boundaries(f, f_new, w, c_x, c_y, boundary_conditions,boundary_values, opposite_i, x, y, width, height);
            });

            // TODO: perhaps separate streaming and boundary handling

            std::swap(f, f_new);

            float amplitude_sum = 0.0f;
            Kokkos::parallel_reduce("sin amplitude",
                Kokkos::MDRangePolicy({0, 0}, {width, height}),
                KOKKOS_LAMBDA(const int x, const int y, float& local_sum) {
                local_sum += v_x(x, y) * sinf(k * static_cast<float>(y));},amplitude_sum);

            measurements_host(state.timestep) =
                amplitude_sum * (1.0f / static_cast<float>(width)) *
                    (1.0f / static_cast<float>(height)) * 2.0f;

            if (state.timestep % state.draw_steps == 0) {
                Kokkos::deep_copy(v_x_host, v_x);
                Kokkos::deep_copy(v_y_host, v_y);

                int frame = state.timestep / state.draw_steps;
                out::write_to("v_x", v_x_host, width, height, frame);
                out::write_to("v_y", v_y_host, width, height, frame);

                std::cout << "Timestep " << state.timestep << "/" << state.max_steps << std::endl;
            }
            state.timestep++;
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

int App::run_mpi(int argc, char *argv[]) const {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        // boundary definition -------------------------------------------------
        int boundary_conditions [2] = {1,1}; // (horizontal, vertical) 0 = periodic, 1 = wall
        float boundary_values [4] = {0.0f, 0.0f, 0.0f, 0.1f}; // speed of wall, ignored if it isn't a wall

        // domain decomposition setup ------------------------------------------
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        /*
        if (size < 2) {
            std::cerr << "Need at least 2 processes" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        */

        int width = 1000;
        int height = 1000;
        int base_n = width / size;
        int remainder = width % size;

        Vec send_buffer_left("send_buffer_left", height * 9);
        Vec send_buffer_right("send_buffer_right", height * 9);
        Vec receive_buffer_left("receive_buffer_left", height * 9);
        Vec receive_buffer_right("receive_buffer_right", height * 9);
        int halo_size = height * 9;

        // Share remainder across the first few ranks
        int rank_width = base_n + (rank < remainder ? 1 : 0);
        int x_offset = rank * base_n + std::min(rank, remainder);

        // Left and right neighbors depend on boundary conditions

        int left = rank - 1;
        int right = rank + 1;
        if (boundary_conditions[0] == 0) {
            left = (left + size) % size;
            right = right % size;
        }else {
            left = rank > 0? left : MPI_PROC_NULL;
            right = rank < size - 1 ? right : MPI_PROC_NULL;
        }

        float local_mass = 0.0f;
        float global_mass = 0.0f;

        // d2q9 initialization -------------------------------------------------
        iVec c_x("c_x", 9);
        iVec c_y("c_y", 9);
        Vec w("w", 9);
        iVec opposite_i("opposite_i", 9);
        auto c_x_host = d2q9::c_x_host_init(c_x);
        auto c_y_host = d2q9::c_y_host_init(c_y);
        auto w_host = d2q9::w_host_init(w);
        auto opposite_i_host = d2q9::opposite_i_host_init(opposite_i);

        // Main view initializations -------------------------------------------
        Density_t rho("rho", rank_width + 2, height);
        Velocity_t v_x("v_x", rank_width + 2, height);
        Velocity_t v_y("v_dty", rank_width + 2, height);
        Distribution_t f("f", rank_width + 2, height, 9);
        Distribution_t f_new("f_new", rank_width + 2, height, 9);

        // Grid initialization -------------------------------------------------
        Kokkos::parallel_for("grid initialization",
            Kokkos::MDRangePolicy({1, 0}, {rank_width + 1, height}),
            KOKKOS_LAMBDA(const int x, const int y) {
            rho(x, y) = 1.0f;
            v_x(x, y) = 0.0f;
            v_y(x, y) = 0.0f;

            for (int i = 0; i < 9; i++) {
                f(x, y, i) = eq::calculate_eq_distrib(
                    rho(x, y), v_x(x, y), v_y(x, y),
                    c_x(i), c_y(i), w(i));
            }
        });
        //Kokkos::fence();
        //dom::exchange_halos(f, send_buffer_left, send_buffer_right, receive_buffer_left, receive_buffer_right, left, right, rank_width, height, halo_size);

        MPI_Barrier(MPI_COMM_WORLD); // Barrier to start timing
        double start_time = MPI_Wtime();

        // Main simulation loop ------------------------------------------------
        for (int step = 0; step < state.max_steps; step++) {

            Kokkos::parallel_for("density + velocity + relaxation step",
                Kokkos::MDRangePolicy({1,0}, {rank_width+1, height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                eq::update_density(rho, f, x, y);
                eq::update_velocity(v_x, v_y, c_x, c_y, rho, f, x, y);
            });

            if (step % state.measurement_interval == 0) {
                Kokkos::parallel_reduce("sum of masses",
                    Kokkos::MDRangePolicy({1,0}, {rank_width+1, height}),
                    KOKKOS_LAMBDA(const int x, const int y, float& sum) {
                    sum += rho(x, y);
                },
                local_mass);
                Kokkos::fence();
                MPI_Reduce(&local_mass, &global_mass, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) {
                    std::cout << "Timestep " << step << "/" << state.max_steps << " | Mass: " << global_mass << std::endl;
                }
            }

            Kokkos::parallel_for("density + velocity + relaxation step",
                Kokkos::MDRangePolicy({1,0}, {rank_width+1, height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                eq::relaxation(f, rho, v_x, v_y, c_x, c_y, w, 1.7f, x, y);
            });


            Kokkos::fence();

            dom::exchange_halos(f, send_buffer_left, send_buffer_right, receive_buffer_left, receive_buffer_right, left, right, rank_width, height, halo_size);

            Kokkos::parallel_for("streaming step",
                Kokkos::MDRangePolicy({1,0}, {rank_width+1, height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                eq::streaming_with_boundaries_mpi(
                    f, f_new, w, c_x, c_y,
                    boundary_conditions, boundary_values, opposite_i,
                    x, x_offset, y, width, height);
            });

            std::swap(f, f_new);
        }

        double end_time = MPI_Wtime();
        double local_elapsed_time = end_time - start_time;
        double global_elapsed_time = 0.0f;

        MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Elapsed time: " << global_elapsed_time << " seconds" << std::endl;
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
