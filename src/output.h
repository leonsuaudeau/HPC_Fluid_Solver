#ifndef HPC_FLUID_SOLVER_OUTPUT_H
#define HPC_FLUID_SOLVER_OUTPUT_H
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <string>

namespace out {
inline constexpr auto base_address = "../../outputs/";
template<class View2D>

void write_to(const std::string& name, const View2D &data, int width, int height, int frame = -1) {
    static_assert(View2D::rank == 2, "write_to expects a rank-2 kokkos view");
    static_assert(std::is_same_v<typename View2D::non_const_value_type, float>,
        "write_to expects a float view");
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace,
        typename View2D::memory_space>::accessible,
        "write_to expects a host-accessible view");

    std::ostringstream filename;
    if (frame == -1) {
        filename << base_address << name << ".bin";
    }else {
        filename << base_address << name << "_" << std::setw(5) << std::setfill('0') << std::to_string(frame) << ".bin";
    }

    std::ofstream file(filename.str(), std::ios::binary);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float value = data(x, y);
            file.write(reinterpret_cast<const char*>(&value), sizeof(float));
        }
    }
    file.close();
}

inline void write_to(const std::string& name, const Kokkos::View<float*, Kokkos::HostSpace>& data) {
    std::ostringstream filename;
    filename << base_address << name << ".bin";
    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<char*>(data.data()), static_cast<long>(data.size() * sizeof(float)));
    file.close();
}
}

#endif // HPC_FLUID_SOLVER_OUTPUT_H
