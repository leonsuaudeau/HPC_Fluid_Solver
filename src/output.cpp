#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include "output.h"

const std::string base_address = "../../outputs/";

void out::write_to(const std::string& name, const Kokkos::View<float **>& data, int width, int height, int frame) {
    std::ostringstream filename;
    filename << base_address << name << "_" << std::setw(5) << std::setfill('0') << std::to_string(frame) << ".bin";
    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<char*>(data.data()), static_cast<long>(width * height * sizeof(float)));
    file.close();
}

void out::write_to(const std::string& name, const Kokkos::View<float*>& data) {
    std::ostringstream filename;
    filename << base_address << name << ".bin";
    std::ofstream file(filename.str(), std::ios::binary);
    file.write(reinterpret_cast<char*>(data.data()), static_cast<long>(data.size() * sizeof(float)));
    file.close();
}
