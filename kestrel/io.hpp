#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <fstream>
#include <map>
#include <string>

class IO {
public:
  static uint32_t read_u32(std::ifstream &f);
  static std::map<std::string, Tensor> read_weights(std::ifstream &f);
};
