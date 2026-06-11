#include "app.h"
#include "d2q9_init.h"
#include "equations.h"
#include "output.h"
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

App::App() : grid(32, 32) {
}

int App::run(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        iVec c_x("c_x", 9);
        iVec c_y("c_y", 9);
        Vec w("w", 9);
        auto c_x_host = d2q9::c_x_host_init(c_x);
        auto c_y_host = d2q9::c_y_host_init(c_y);
        auto w_host = d2q9::w_host_init(w);

        Density_t rho("rho", grid.width, grid.height);
        Velocity_t v_x("v_x", grid.width, grid.height);
        Velocity_t v_y("v_dty", grid.width, grid.height);
        Distribution_t f("f", grid.width, grid.height, 9);
        Distribution_t f_new("f_new", grid.width, grid.height, 9);

        auto rho_host = Kokkos::create_mirror_view(rho);
        auto v_x_host = Kokkos::create_mirror_view(v_x);
        auto v_y_host = Kokkos::create_mirror_view(v_y);
        auto f_host = Kokkos::create_mirror_view(f);

        Vec measurements("m", state.max_steps);
        auto measurements_host = Kokkos::create_mirror_view(measurements);

        for (int o = 1; o <= 20; o++) {

            // Grid initializations
            float n_y_inv = 1.0f / static_cast<float>(grid.height);
            float k = 2.0f * 3.14159265359f * n_y_inv;
            for (int x = 0; x < grid.width; x++) {
                for (int y = 0; y < grid.height; y++) {
                    rho_host(x, y) = 0.5f;
                    v_x_host(x, y) = eq::sinusoidal(y, 1.0f, k);
                    v_y_host(x, y) = 0.0f;

                    /*
                    if (x == 15 && y == 15) {
                        rho_host(x, y) = 10.0f;
                    }
                    */

                    for (int i = 0; i < 9; i++) {
                        f_host(x, y, i) = eq::calculate_eq_distrib(rho_host(x, y), v_x_host(x, y), v_y_host(x, y), c_x_host(i), c_y_host(i), w_host(i));
                    }
                }
            }

            Kokkos::deep_copy(rho, rho_host);
            Kokkos::deep_copy(v_x, v_x_host);
            Kokkos::deep_copy(v_y, v_y_host);
            Kokkos::deep_copy(f, f_host);

            while (state.timestep < state.max_steps) {
                // Simulation step
                Kokkos::parallel_for("density + velocity + relaxation step",
                    Kokkos::MDRangePolicy({0,0}, {grid.width, grid.height}),
                    KOKKOS_LAMBDA(const int x, const int y) {
                    eq::update_density(rho, f, x, y);
                    eq::update_velocity(v_x, v_y, c_x, c_y, rho, f, x, y);
                    const float omega = 1.0f / static_cast<float>(o) * 2.0f;
                    eq::relaxation(f, rho, v_x, v_y, c_x, c_y, w, omega, x, y);
                });

                Kokkos::parallel_for("streaming step",
                    Kokkos::MDRangePolicy({0,0}, {grid.width, grid.height}),
                    KOKKOS_LAMBDA(const int x, const int y) {
                    eq::streaming(f, f_new, c_x, c_y, x, y, grid.width, grid.height);
                });

                std::swap(f, f_new);

                measurements[state.timestep] = eq::calculate_sin_amplitude(v_x, v_y, k);

                /*
                if (state.timestep % state.draw_steps == 0) {
                    Kokkos::deep_copy(v_x_host, v_x);
                    Kokkos::deep_copy(v_y_host, v_y);

                    int frame = state.timestep / state.draw_steps;
                    out::write_to("v_x", v_x_host, grid.width, grid.height, frame);
                    out::write_to("v_y", v_y_host, grid.width, grid.height, frame);

                    std::cout << "Timestep " << state.timestep << "/" << state.max_steps << std::endl;
                }
                */
                state.timestep++;
            }
            state.timestep = 0;
            std::ostringstream name;
            name << "measurements" << "_" << std::setw(2) << std::setfill('0') << std::to_string(o);
            Kokkos::deep_copy(measurements_host, measurements);
            out::write_to(name.str(), measurements_host);
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
