#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include "output.h"

void out::write_to(std::string name, Kokkos::View<float **> data, int width, int height, int frame) {
    std::ostringstream filename;
    filename << "../../outputs/" << name << "_" << std::setw(5) << std::setfill('0') << std::to_string(frame) << ".bin";
    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<char*>(data.data()), static_cast<long>(width * height * sizeof(float)));
    file.close();
}