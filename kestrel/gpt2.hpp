#pragma once

#include "config.hpp"
#include "kv_cache.hpp"
#include "tensor.hpp"
#include <map>
#include <string>
#include <vector>

class GPT2 {
public:
  explicit GPT2(std::map<std::string, Tensor> weights,
                const GPT2Config &config);

  const GPT2Config &config() const { return config_; }

  std::vector<float> forward(const std::vector<int> &tokens, KVCache &cache,
                             bool all_logits = false) const;
  std::vector<float> step(int token, int pos, KVCache &cache) const;
  std::vector<float> prefill(const std::vector<int> &tokens,
                             KVCache &cache) const;

private:
  std::map<std::string, Tensor> weights;
  std::map<std::string, QTensor> qweights;
  GPT2Config config_;

  const QTensor &qparam(const std::string &name) const {
    return qweights.at(name);
  }

  const QTensor &qparam(int layer, const std::string &name) const {
    return qparam("h." + std::to_string(layer) + "." + name);
  }

  const float *param(const std::string &name) const {
    return weights.at(name).data.data();
  }

  const float *param(int layer, const std::string &name) const {
    return param("h." + std::to_string(layer) + "." + name);
  }

  void transformer_block(int layer, std::vector<float> &x, int T, int start_pos,
                         KVCache &cache) const;
};
