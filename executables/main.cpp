#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

KOKKOS_INLINE_FUNCTION
int wrap(const int i, const int m) {
    return (i % m + m) % m;
}

KOKKOS_INLINE_FUNCTION
void update_density(
    const Kokkos::View<float**> &rho,
    const Kokkos::View<float***> &f,
    const int x, const int y)
{
    float sum = 0.0f;
    for (int i = 0; i < 9; i++) {
        sum += f(x, y, i);
    }
    rho(x, y) = sum;
}

KOKKOS_INLINE_FUNCTION
void update_velocity(
    const Kokkos::View<float**> &v_x,
    const Kokkos::View<float**> &v_y,
    const Kokkos::View<int*> &c_x,
    const Kokkos::View<int*> &c_y,
    const Kokkos::View<float**> &rho,
    const Kokkos::View<float***> &f,
    const int x, const int y)
{
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
    const Kokkos::View<float***> &f,
    const Kokkos::View<float***> &f_new,
    const Kokkos::View<int*> &c_x,
    const Kokkos::View<int*> &c_y,
    const int x, const int y, const int grid_width, const int grid_height)
{
    for (int i = 0; i < 9; i++) {
        const int x_old = wrap(x - c_x(i), grid_width);
        const int y_old = wrap(y - c_y(i), grid_height);
        f_new(x, y, i) = f(x_old, y_old, i);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        int grid_width, grid_height;
        grid_width = grid_height = 32;
        int timestep = 0;
        int max_steps = 100;
        int draw_steps = 1;

        Kokkos::View<int*> c_x("c_x", 9);
        Kokkos::View<int*> c_y("c_y", 9);
        auto c_x_host = Kokkos::create_mirror_view(c_x);
        auto c_y_host = Kokkos::create_mirror_view(c_y);

        // D2Q9 scheme initialization
        c_x_host(0) = 0.0f; c_y_host(0) = 0.0f;
        c_x_host(1) = 0.0f; c_y_host(1) = 1.0f;
        c_x_host(2) = 1.0f; c_y_host(2) = 0.0f;
        c_x_host(3) = 0.0f; c_y_host(3) = -1.0f;
        c_x_host(4) = -1.0f; c_y_host(4) = 0.0f;
        c_x_host(5) = 1.0f; c_y_host(5) = 1.0f;
        c_x_host(6) = 1.0f; c_y_host(6) = -1.0f;
        c_x_host(7) = -1.0f; c_y_host(7) = -1.0f;
        c_x_host(8) = -1.0f; c_y_host(8) = 1.0f;

        Kokkos::deep_copy(c_x, c_x_host);
        Kokkos::deep_copy(c_y, c_y_host);

        Kokkos::View<float**> rho("rho", grid_width, grid_height);
        Kokkos::View<float**> v_x("v_x", grid_width, grid_height);
        Kokkos::View<float**> v_y("v_y", grid_width, grid_height);
        Kokkos::View<float***> f("f", grid_width, grid_height, 9);
        Kokkos::View<float***> f_new("f_new", grid_width, grid_height, 9);

        auto rho_host = Kokkos::create_mirror_view(rho);
        auto v_x_host = Kokkos::create_mirror_view(v_x);
        auto v_y_host = Kokkos::create_mirror_view(v_y);
        auto f_host = Kokkos::create_mirror_view(f);

        // Grid initializations
        for (int x = 0; x < grid_width; x++) {
            for (int y = 0; y < grid_height; y++) {
                rho_host(x, y) = 1.0f;
                v_x_host(x, y) = 0.0f; // small flow for testing
                v_y_host(x, y) = 0.0f;
                for (int i = 0; i < 9; i++) {
                    if (x == 31 && i == 4) {
                        f_host(x, y, i) = 1.0f;
                        continue;
                    }
                    f_host(x, y, i) = 0.01f; // static_cast<float>(random());
                }
            }
        }

        Kokkos::deep_copy(rho, rho_host);
        Kokkos::deep_copy(v_x, v_x_host);
        Kokkos::deep_copy(v_y, v_y_host);
        Kokkos::deep_copy(f, f_host);


        while (timestep < max_steps) {
            // Simulation step
            Kokkos::parallel_for("update_density",
                Kokkos::MDRangePolicy({0,0}, {grid_width, grid_height}),
                KOKKOS_LAMBDA(const int x, const int y) {
                    update_density(rho, f, x, y);
                    update_velocity(v_x, v_y, c_x, c_y, rho, f, x, y);
                    streaming(f, f_new, c_x, c_y, x, y, grid_width, grid_height);
                }
            );
            Kokkos::deep_copy(f, f_new);

            if (timestep % draw_steps == 0) {
                Kokkos::deep_copy(v_x_host, v_x);
                Kokkos::deep_copy(v_y_host, v_y);

                std::ostringstream filename;
                filename << "../../outputs/v_x_" << std::setw(5) << std::setfill('0') << std::to_string(timestep / draw_steps) << ".bin";
                std::ofstream file(filename.str(), std::ios::binary);
                file.write(reinterpret_cast<char*>(v_x_host.data()), grid_width * grid_height * sizeof(float));
                file.close();

                std::ostringstream filename2;
                filename2 << "../../outputs/v_y_" << std::setw(5) << std::setfill('0') << std::to_string(timestep / draw_steps) << ".bin";
                std::ofstream file2(filename2.str(), std::ios::binary);
                file2.write(reinterpret_cast<char*>(v_y_host.data()), grid_width * grid_height * sizeof(float));
                file2.close();

                std::cout << "Timestep " << timestep << "/" << max_steps << std::endl;
            }
            timestep++;
        }
    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
