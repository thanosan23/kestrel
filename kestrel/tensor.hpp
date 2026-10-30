#pragma once
#include <vector>

struct Tensor {
  std::vector<uint32_t> shape;
  std::vector<float> data;
};
