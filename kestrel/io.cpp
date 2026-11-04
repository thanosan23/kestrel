#include "io.hpp"
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "tensor.hpp"

uint32_t IO::read_u32(std::ifstream &f) {
  uint32_t v;
  f.read((char *)&v, 4);
  return v;
}

std::map<std::string, Tensor> IO::read_weights(std::ifstream &f) {
  std::map<std::string, Tensor> weights;
  uint32_t count = IO::read_u32(f);
  for (uint32_t i = 0; i < count; i++) {
    std::string name(IO::read_u32(f), '\0');
    f.read(name.data(), name.size());

    Tensor t;
    t.shape.resize(IO::read_u32(f));
    uint32_t size = 1;
    for (auto &dim : t.shape) {
      dim = IO::read_u32(f);
      size *= dim;
    }
    t.data.resize(size);
    f.read((char *)t.data.data(), size * sizeof(float));
    weights[name] = std::move(t);
  }
  return weights;
}
