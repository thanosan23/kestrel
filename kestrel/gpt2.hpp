#pragma once

#include "kv_cache.hpp"
#include "tensor.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

class GPT2 {
public:
  explicit GPT2(std::map<std::string, Tensor> weights)
      : weights(std::move(weights)) {};

  std::vector<float> step(int token, int pos, KVCache &cache) const;

private:
  std::map<std::string, Tensor> weights;

  const float *param(const std::string &name) const {
    return weights.at(name).data.data();
  }

  const float *param(int layer, const std::string &name) const {
    return param("h." + std::to_string(layer) + "." + name);
  }

  void transformer_block(int layer, std::vector<float> &x, int pos,
                         KVCache &cache) const;
};
