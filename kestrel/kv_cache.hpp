#pragma once

#include "config.hpp"
#include <vector>

struct KVCache {
  std::vector<std::vector<float>> k, v;
  KVCache()
      : k(cfg::n_layer, std::vector<float>(cfg::n_ctx * cfg::C)),
        v(cfg::n_layer, std::vector<float>(cfg::n_ctx * cfg::C)) {}
};
