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
    Kokkos::initialize(argc, argv);
    {
        int width = 64;
        int height = 64;
        float omega = 1.7f;

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

        Kokkos::Array<int, 2> boundary_conditions = {0, 0}; // (horizontal, vertical) 0 = periodic, 1 = wall
        Kokkos::Array<float, 4> boundary_values = {0.0f, 0.0f, 0.0f, 0.1f}; // speed of wall, ignored if it isn't a wall
        // TODO: fix corners, use 2 vel. values per wall for this!

        auto rho_host = Kokkos::create_mirror_view(rho);
        auto v_x_host = Kokkos::create_mirror_view(v_x);
        auto v_y_host = Kokkos::create_mirror_view(v_y);
        auto f_host = Kokkos::create_mirror_view(f);

        Vec_host measurements_host("m", state.max_steps);

        for (int i = 1; i < 100; i++) {
            omega = static_cast<float>(i) / 50.0f;
            std::cout << "Omega: " << omega << std::endl;

            // Grid initializations
            float n_y_inv = 1.0f / static_cast<float>(height);
            float k = 2.0f * 3.14159265359f * n_y_inv;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    rho_host(x, y) = 1.0f;
                    v_x_host(x, y) = eq::sinusoidal(y, 0.1f, k);
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

            double global_mass = 0.0f;
            double global_kinetic_energy = 0.0f;

            std::cout << "Reynolds number: " << eq::calculate_reynolds_number(boundary_values, width, omega) << std::endl;

            while (state.timestep < state.max_steps) {
                // Simulation step
                Kokkos::parallel_for("one-step pull scheme step",
                    Kokkos::MDRangePolicy({0,0}, {width, height}),
                    KOKKOS_LAMBDA(const int x, const int y) {
                    eq::streaming_with_boundaries(f, f_new, w, c_x, c_y, boundary_conditions,boundary_values, opposite_i, x, y, width, height);
                    eq::update_density(rho, f_new, x, y);
                    eq::update_velocity(v_x, v_y, c_x, c_y, rho, f_new, x, y);
                    eq::relaxation(f_new, rho, v_x, v_y, c_x, c_y, w, omega, x, y);
                });
                std::swap(f, f_new);

                // Now measure some values for testing
                if (state.timestep % state.measurement_interval == 0) {
                    Kokkos::parallel_reduce("sum of masses",
                        Kokkos::MDRangePolicy({0,0}, {width, height}),
                        KOKKOS_LAMBDA(const int x, const int y, double& local_mass) {
                        local_mass += rho(x, y);
                    },
                    global_mass);

                    Kokkos::parallel_reduce("sum of kinetic energies",
                        Kokkos::MDRangePolicy({0,0}, {width, height}),
                        KOKKOS_LAMBDA(const int x, const int y, double& local_energy) {
                        local_energy += 0.5 * rho(x, y) * v_x(x, y) * v_x(x, y) * v_y(x, y) * v_y(x, y);
                    },
                    global_kinetic_energy);

                    std::cout << "Timestep " << state.timestep << "/" << state.max_steps << " | Mass: " << global_mass << " | Kinetic Energy: " << global_kinetic_energy << std::endl;
                }

                float amplitude_sum = 0.0f;
                Kokkos::parallel_reduce("sin amplitude",
                    Kokkos::MDRangePolicy({0, 0}, {width, height}),
                    KOKKOS_LAMBDA(const int x, const int y, float& local_sum) {
                    local_sum += v_x(x, y) * sinf(k * static_cast<float>(y));},amplitude_sum);

                measurements_host(state.timestep) =
                    amplitude_sum * (1.0f / static_cast<float>(width)) *
                        (1.0f / static_cast<float>(height)) * 2.0f;

                /*
                if (state.timestep % state.draw_steps == 0) {
                    Kokkos::deep_copy(v_x_host, v_x);
                    Kokkos::deep_copy(v_y_host, v_y);

                    int frame = state.timestep / state.draw_steps;
                    out::write_to("v_x", v_x_host, width, height, frame);
                    out::write_to("v_y", v_y_host, width, height, frame);

                    std::cout << "Timestep " << state.timestep << "/" << state.max_steps << std::endl;
                }
                */
                state.timestep++;
            }
            state.timestep = 0;
            std::ostringstream filename;
            filename << "shear_wave/different_k/amplitude_" << std::setw(3) << std::setfill('0') << std::to_string(i);
            out::write_to(filename.str(), measurements_host);
        }

        // For moving lid plotting, comment out the other exports and uncomment this
        // You should plot different raynolds numbers, grid sizes, etc.
        /*

        Kokkos::deep_copy(v_y_host, v_y);

        out::write_to("moving_lid/v_x", v_x_host, width, height);
        out::write_to("moving_lid/v_y", v_y_host, width, height);

        std::cout << "Timestep " << state.timestep << "/" << state.max_steps << std::endl;
        */
        //out::write_to("shear_wave/amplitude", measurements_host);
    }

    Kokkos::finalize();

    return 0;
}

int App::run_mpi_strips(int argc, char *argv[]) const {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    {
        // boundary definition -------------------------------------------------
        Kokkos::Array<int, 2> boundary_conditions = {1, 1}; // (horizontal, vertical) 0 = periodic, 1 = wall
        Kokkos::Array<float, 4> boundary_values = {0.0f, 0.0f, 0.0f, 0.1f}; // speed of wall, ignored if it isn't a wall

        // domain decomposition setup ------------------------------------------
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int grid_width = 4096;
        int grid_height = 4096;
        int base_strip_width = grid_width / size;
        int remainder = grid_width % size;

        Vec send_buffer_left("send_buffer_left", grid_height * 9);
        Vec send_buffer_right("send_buffer_right", grid_height * 9);
        Vec receive_buffer_left("receive_buffer_left", grid_height * 9);
        Vec receive_buffer_right("receive_buffer_right", grid_height * 9);
        int halo_size = grid_height * 9;

        // Share remainder across the first few ranks
        int strip_width = base_strip_width + (rank < remainder ? 1 : 0);
        int x_offset = rank * base_strip_width + std::min(rank, remainder);

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

        double local_mass = 0.0;
        double global_mass = 0.0;
        double local_kinetic_energy = 0.0;
        double global_kinetic_energy = 0.0;

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
        Density_t rho("rho", strip_width + 2, grid_height);
        Velocity_t v_x("v_x", strip_width + 2, grid_height);
        Velocity_t v_y("v_dty", strip_width + 2, grid_height);
        Distribution_t f("f", strip_width + 2, grid_height, 9);
        Distribution_t f_new("f_new", strip_width + 2, grid_height, 9);

        // Grid initialization -------------------------------------------------
        Kokkos::parallel_for("grid initialization",
            Kokkos::MDRangePolicy({1, 0}, {strip_width + 1, grid_height}),
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
        // exchange once before loop for valid halos for first streaming step
        Kokkos::fence();
        dom::exchange_halos_vert_strips(f,
            send_buffer_left, send_buffer_right,
            receive_buffer_left, receive_buffer_right,
            left, right, strip_width, grid_height, halo_size);

        MPI_Barrier(MPI_COMM_WORLD); // Barrier to start timing
        double start_time = MPI_Wtime();

        // Main simulation loop ------------------------------------------------
        for (int step = 0; step < state.max_steps; step++) {
            Kokkos::parallel_for("one-step pull scheme step",
                    Kokkos::MDRangePolicy({1,0}, {strip_width+1, grid_height}),
                    KOKKOS_LAMBDA(const int x, const int y) {
                    eq::streaming_with_boundaries_mpi_strips(
                        f, f_new, w, c_x, c_y,
                        boundary_conditions, boundary_values, opposite_i,
                        x, x_offset, y, grid_width, grid_height);
                    eq::update_density(rho, f_new, x, y);
                    eq::update_velocity(v_x, v_y, c_x, c_y, rho, f_new, x, y);
                    eq::relaxation(f_new, rho, v_x, v_y, c_x, c_y, w, 1.7f, x, y);
                });

            Kokkos::fence();
            dom::exchange_halos_vert_strips(f_new,
                send_buffer_left, send_buffer_right,
                receive_buffer_left, receive_buffer_right,
                left, right, strip_width, grid_height, halo_size);

            std::swap(f, f_new);

            if (step % state.measurement_interval == 0) {
                Kokkos::parallel_reduce("sum of masses",
                    Kokkos::MDRangePolicy({1,0}, {strip_width+1, grid_height}),
                    KOKKOS_LAMBDA(const int x, const int y, double& sum) {
                    sum += rho(x, y);
                },local_mass);
                Kokkos::parallel_reduce("sum of kinetic energies",
                    Kokkos::MDRangePolicy({1,0}, {strip_width+1, grid_height}),
                    KOKKOS_LAMBDA(const int x, const int y, double& sum) {
                    sum += 0.5 * rho(x, y) * (v_x(x, y) + v_x(x, y)) * (v_y(x, y) + v_y(x, y));
                },local_kinetic_energy);
                Kokkos::fence();

                MPI_Reduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&local_kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (rank == 0) {
                    std::cout << "Timestep " << step << "/" << state.max_steps << " | Mass: " << global_mass << " | Kinetic Energy: " << global_kinetic_energy << std::endl;
                }
            }
        }

        double end_time = MPI_Wtime();
        double local_elapsed_time = end_time - start_time;
        double global_elapsed_time = 0.0f;

        MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Elapsed time: " << global_elapsed_time << " seconds" << std::endl;
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

int App::run_mpi_tiles(int argc, char *argv[]) const {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        // boundary definition -------------------------------------------------
        Kokkos::Array<int, 2> boundary_conditions = {1, 1}; // (horizontal, vertical) 0 = periodic, 1 = wall
        Kokkos::Array<float, 4> boundary_values = {0.0f, 0.0f, 0.0f, 0.1f}; // speed of wall, ignored if it isn't a wall

        // domain decomposition setup ------------------------------------------
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int dims[2] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        int periods[2];
        periods[0] = boundary_conditions[0] == 0;
        periods[1] = boundary_conditions[1] == 0;

        MPI_Comm cart;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

        int rank, coords[2];
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 2, coords);

        if (rank == 0) {
            Kokkos::print_configuration(std::cout);
        }

        int grid_width = 30000;
        int grid_height = 30000;
        int base_tile_width = grid_width / dims[0];
        int base_tile_height = grid_height / dims[1];
        int remainder_x = grid_width % dims[0];
        int remainder_y = grid_height % dims[1];

        int tile_width = base_tile_width + (coords[0] < remainder_x ? 1 : 0);
        int tile_height = base_tile_height + (coords[1] < remainder_y ? 1 : 0);
        int x_offset = coords[0] * base_tile_width + std::min(coords[0], remainder_x);
        int y_offset = coords[1] * base_tile_height + std::min(coords[1], remainder_y);

        Vec send_buffer_left("send_buffer_left", tile_height * 3);
        Vec send_buffer_right("send_buffer_right", tile_height * 3);
        Vec receive_buffer_left("receive_buffer_left", tile_height * 3);
        Vec receive_buffer_right("receive_buffer_right", tile_height * 3);

        Vec send_buffer_up("send_buffer_up", (tile_width + 2) * 3);
        Vec send_buffer_down("send_buffer_down", (tile_width + 2) * 3);
        Vec receive_buffer_up("receive_buffer_up", (tile_width + 2) * 3);
        Vec receive_buffer_down("receive_buffer_down", (tile_width + 2) * 3);

        const int halo_size[2] = {tile_height * 3, (tile_width + 2) * 3};

        int left, right, up, down;
        MPI_Cart_shift(cart, 0, 1, &left, &right);
        MPI_Cart_shift(cart, 1, 1, &down, &up);

        double local_mass = 0.0;
        double global_mass = 0.0;
        double local_kinetic_energy = 0.0;
        double global_kinetic_energy = 0.0;

        // d2q9 initialization -------------------------------------------------
        iVec c_x("c_x", 9);
        iVec c_y("c_y", 9);
        Vec w("w", 9);
        iVec opposite_i("opposite_i", 9);
        auto c_x_host = d2q9::c_x_host_init(c_x);
        auto c_y_host = d2q9::c_y_host_init(c_y);
        auto w_host = d2q9::w_host_init(w);
        auto opposite_i_host = d2q9::opposite_i_host_init(opposite_i);

        Kokkos::Array<int, 3> move_left_dirs = {3,6,7};
        Kokkos::Array<int, 3> move_right_dirs = {1,5,8};
        Kokkos::Array<int, 3> move_up_dirs = {2,5,6};
        Kokkos::Array<int, 3> move_down_dirs = {4,7,8};

        // Main view initializations -------------------------------------------
        Distribution_t_flat f("f", (tile_width + 2) * (tile_height + 2) * 9);
        Distribution_t_flat f_new("f_new", (tile_width + 2) * (tile_height + 2) * 9);

        // Grid initialization -------------------------------------------------
        Kokkos::parallel_for("grid initialization",
        Kokkos::MDRangePolicy<Left_Policy>({1, 1}, {tile_width + 1, tile_height + 1}),
            KOKKOS_LAMBDA(const int x, const int y) {
                for (int i = 0; i < 9; i++) {
                    constexpr float u_y = 0.0f;
                    constexpr float u_x = 0.0f;
                    constexpr float rho = 1.0f;
                    const int idx = x + (tile_width + 2) * (y + (tile_height + 2) * i);
                    f(idx) = eq::calculate_eq_distrib(
                    rho, u_x, u_y,
                    c_x(i), c_y(i), w(i));
            }
        });
        Kokkos::fence();
        // 2-pass exchange, first horizontal, then vertical including ghost corners
        dom::exchange_halos_tiles_x_pass(f, cart,
            send_buffer_left, send_buffer_right,
            receive_buffer_left, receive_buffer_right,
            left, right, tile_width, tile_height,
            halo_size, move_left_dirs, move_right_dirs);
        dom::exchange_halos_tiles_y_pass(f, cart,
            send_buffer_up, send_buffer_down,
            receive_buffer_up, receive_buffer_down,
            up, down, tile_width, tile_height,
            halo_size, move_up_dirs, move_down_dirs);

        const int boundary_count = 2 * tile_width + 2 * (tile_height - 2);

        MPI_Barrier(MPI_COMM_WORLD); // Barrier to start timing
        double start_time = MPI_Wtime();

        // Main simulation loop ------------------------------------------------
        for (int step = 0; step < state.max_steps; step++) {

            Kokkos::parallel_for("fast interior update",
                Kokkos::MDRangePolicy<Left_Policy>({2,2}, {tile_width, tile_height}),
                KOKKOS_LAMBDA(const int x, const int y) {

                eq::main_kernel_interior(f, f_new, x, y, tile_width, tile_height, 1.7f);
            });

            Kokkos::parallel_for("branching boundary update",
                Kokkos::RangePolicy(0, boundary_count),
                KOKKOS_LAMBDA(const int n) {
                int x; int y;

                if (n < tile_width) {
                    x = n + 1; y = 1;
                }else  if (n < 2 * tile_width) {
                    x = n - tile_width + 1;
                    y = tile_height;
                }else if (n < 2 * tile_width + (tile_height - 2)) {
                    x = 1;
                    y = n - 2 * tile_width + 2;
                }else {
                    x = tile_width;
                    y = n - (2 * tile_width + (tile_height - 2)) + 2;
                }

                eq::main_kernel_boundary(f, f_new, boundary_conditions, boundary_values,
                    x, y, x_offset, y_offset, grid_width, grid_height, tile_width, tile_height, 1.7f);
            });

            Kokkos::fence();

            dom::exchange_halos_tiles_x_pass(f_new, cart,
                send_buffer_left, send_buffer_right,
                receive_buffer_left, receive_buffer_right,
                left, right, tile_width, tile_height,
                halo_size, move_left_dirs, move_right_dirs);
            dom::exchange_halos_tiles_y_pass(f_new, cart,
                send_buffer_up, send_buffer_down,
                receive_buffer_up, receive_buffer_down,
                up, down, tile_width, tile_height,
                halo_size, move_up_dirs, move_down_dirs);

            std::swap(f, f_new);

            /*

            if (step % state.measurement_interval == 0) {
                const int width_with_halos = tile_width + 2;
                const int offset = width_with_halos * (tile_height + 2);

                Kokkos::parallel_reduce("sum of masses",
                    Kokkos::MDRangePolicy<Left_Policy>({1,1}, {tile_width+1, tile_height+1}),
                    KOKKOS_LAMBDA(const int x, const int y, double& sum) {
                    const int base = x + width_with_halos * y;

                    const float rho_cell =
                        f(base) +
                        f(base + offset) +
                        f(base + 2 * offset) +
                        f(base + 3 * offset) +
                        f(base + 4 * offset) +
                        f(base + 5 * offset) +
                        f(base + 6 * offset) +
                        f(base + 7 * offset) +
                        f(base + 8 * offset);
                    sum += rho_cell;
                },local_mass);
                Kokkos::parallel_reduce("sum of kinetic energies",
                    Kokkos::MDRangePolicy<Left_Policy>({1,1}, {tile_width+1, tile_height+1}),
                    KOKKOS_LAMBDA(const int x, const int y, double& sum) {
                    const int base = x + width_with_halos * y;

                    const float f_0 = f(base);
                    const float f_1 = f(base + offset);
                    const float f_2 = f(base + 2 * offset);
                    const float f_3 = f(base + 3 * offset);
                    const float f_4 = f(base + 4 * offset);
                    const float f_5 = f(base + 5 * offset);
                    const float f_6 = f(base + 6 * offset);
                    const float f_7 = f(base + 7 * offset);
                    const float f_8 = f(base + 8 * offset);

                    const float rho = f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8;
                    const float u_x = (f_1 - f_3 + f_5 - f_6 - f_7 + f_8) / rho;
                    const float u_y = (f_2 - f_4 + f_5 + f_6 - f_7 - f_8) / rho;

                    sum += 0.5 * rho * (u_x * u_x + u_y * u_y);
                },local_kinetic_energy);
                Kokkos::fence();

                MPI_Reduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
                MPI_Reduce(&local_kinetic_energy, &global_kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
                if (rank == 0) {
                    std::cout << "Timestep " << step << "/" << state.max_steps << " | Mass: " << global_mass << " | Kinetic Energy: " << global_kinetic_energy << std::endl;
                }
            }
            */
        }

        double end_time = MPI_Wtime();
        double local_elapsed_time = end_time - start_time;
        double global_elapsed_time = 0.0f;

        MPI_Reduce(&local_elapsed_time, &global_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
        if (rank == 0) {
            std::cout << "Elapsed time: " << global_elapsed_time << " seconds" << std::endl;
            std::cout << "BLUPS: " << eq::blups(grid_width, grid_height, state.max_steps, global_elapsed_time);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
