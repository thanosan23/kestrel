#pragma once

#include "config.hpp"
#include <stdexcept>
#include <vector>

struct KVCache {
  std::vector<std::vector<float>> k, v;
  int len = 0;

  explicit KVCache(const GPT2Config &c)
      : k(c.n_layer, std::vector<float>((size_t)GPT2Config::n_ctx * c.C)),
        v(c.n_layer, std::vector<float>((size_t)GPT2Config::n_ctx * c.C)) {}

  void truncate(int n) {
    if (n < 0 || n > len) {
      throw std::runtime_error("truncate: n must be between 0 and len");
    }
    len = n;
  }
};
