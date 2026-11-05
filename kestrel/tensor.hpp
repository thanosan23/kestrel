#pragma once
#include <vector>

struct Tensor {
  std::vector<uint32_t> shape;
  std::vector<float> data;
};

struct QTensor {
  std::vector<int8_t> q;
  std::vector<float> scale;
  int rows = 0, cols = 0;
};
