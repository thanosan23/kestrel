#include "gpt2.hpp"
#include "config.hpp"
#include "kv_cache.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include "threadpool.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void attention(const GPT2Config &cfg, float *out, const float *q,
                      const float *K, const float *V, int pos) {
  constexpr int head_dim = GPT2Config::head_dim;
  const int C = cfg.C;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  std::vector<float> probs(pos + 1);

  for (int h = 0; h < cfg.n_head; h++) {
    const float *qh = q + h * head_dim;
    float max_score = -INFINITY;
    for (int t2 = 0; t2 <= pos; t2++) {
      const float *kh = K + t2 * C + h * head_dim;
      probs[t2] = dot(qh, kh, head_dim) * scale;
      max_score = std::max(max_score, probs[t2]);
    }

    float sum = 0.0f;
    for (int t2 = 0; t2 <= pos; t2++) {
      probs[t2] = std::exp(probs[t2] - max_score);
      sum += probs[t2];
    }
    for (int t2 = 0; t2 <= pos; t2++) {
      probs[t2] /= sum;
    }

    float *o = out + h * head_dim;
    std::fill(o, o + head_dim, 0.0f);
    for (int t2 = 0; t2 <= pos; t2++) {
      const float *vh = V + t2 * C + h * head_dim;
      for (int i = 0; i < head_dim; i++) {
        o[i] += probs[t2] * vh[i];
      }
    }
  }
}

GPT2::GPT2(std::map<std::string, Tensor> w, const GPT2Config &config)
    : weights(std::move(w)), config_(config) {
  const int C = config_.C;
  const int n_layer = config_.n_layer;

  const std::string last = "h." + std::to_string(n_layer - 1) + ".ln_1.weight";
  const std::string extra = "h." + std::to_string(n_layer) + ".ln_1.weight";
  if (weights.at("wpe.weight").data.size() != (size_t)GPT2Config::n_ctx * C ||
      !weights.count(last) || weights.count(extra)) {
    throw std::runtime_error("weights don't match the GPT2Config");
  }

  auto quantize = [&](const std::string &name, int rows, int cols) {
    qweights[name] = quantize_rows(weights.at(name).data, rows, cols);
    weights.erase(name);
  };

  for (int layer = 0; layer < n_layer; layer++) {
    std::string p = "h." + std::to_string(layer) + ".";
    quantize(p + "attn.c_attn.weight", C, 3 * C);
    quantize(p + "attn.c_proj.weight", C, C);
    quantize(p + "mlp.c_fc.weight", C, 4 * C);
    quantize(p + "mlp.c_proj.weight", 4 * C, C);
  }
  quantize("wte.weight", GPT2Config::V, C);
}

std::vector<float> GPT2::forward(const std::vector<int> &tokens, KVCache &cache,
                                 bool all_logits) const {
  constexpr int V = GPT2Config::V;
  const int C = config_.C;
  const int T = tokens.size();
  const int start_pos = cache.len;

  if (T == 0)
    throw std::runtime_error("forward: no tokens");

  if (start_pos + T > GPT2Config::n_ctx) {
    throw std::runtime_error("context is full (" +
                             std::to_string(GPT2Config::n_ctx) + " tokens)");
  }

  const QTensor &wte = qparam("wte.weight");
  const float *wpe = param("wpe.weight");

  std::vector<float> x((size_t)T * C);
  for (int t = 0; t < T; t++) {
    const int8_t *tok_row = wte.q.data() + tokens[t] * C;
    const float s = wte.scale[tokens[t]];
    for (int c = 0; c < C; c++) {
      x[(size_t)t * C + c] =
          tok_row[c] * s + wpe[(size_t)(t + start_pos) * C + c];
    }
  }
  for (int layer = 0; layer < config_.n_layer; layer++) {
    transformer_block(layer, x, T, start_pos, cache);
  }
  cache.len = start_pos + T;

  const int first = all_logits ? 0 : T - 1;
  const int R = T - first;

  std::vector<float> normed(R * C);
  layernorm(normed.data(), x.data() + (size_t)first * C, param("ln_f.weight"),
            param("ln_f.bias"), R, C);

  std::vector<float> logits(R * V);
  parallel_for(V, [&](int v0, int v1) {
    for (int v = v0; v < v1; v++) {
      const int8_t *row = wte.q.data() + (size_t)v * C;
      const float s = wte.scale[v];
      for (int r = 0; r < R; r++) {
        logits[(size_t)r * V + v] =
            s * dot_q(normed.data() + (size_t)r * C, row, C);
      }
    }
  });
  return logits;
}

std::vector<float> GPT2::prefill(const std::vector<int> &tokens,
                                 KVCache &cache) const {
  cache.truncate(0);
  return forward(tokens, cache);
}

std::vector<float> GPT2::step(int token, int pos, KVCache &cache) const {
  if (pos != cache.len) {
    throw std::runtime_error("step: pos must equal cache.len");
  }
  return forward({token}, cache);
}

void GPT2::transformer_block(int layer, std::vector<float> &x, int T,
                             int start_pos, KVCache &cache) const {
  const int C = config_.C;

  std::vector<float> normed((size_t)T * C), qkv((size_t)T * 3 * C),
      attn((size_t)T * C), hidden((size_t)T * 4 * C), out((size_t)T * C);

  layernorm(normed.data(), x.data(), param(layer, "ln_1.weight"),
            param(layer, "ln_1.bias"), T, C);

  linear_q(qkv.data(), normed.data(), qparam(layer, "attn.c_attn.weight"),
           param(layer, "attn.c_attn.bias"), T, C, 3 * C);

  float *K = cache.k[layer].data();
  float *V = cache.v[layer].data();

  for (int t = 0; t < T; t++) {
    const float *row = qkv.data() + (size_t)t * 3 * C;
    std::copy(row + C, row + 2 * C, K + (size_t)(start_pos + t) * C);
    std::copy(row + 2 * C, row + 3 * C, V + (size_t)(start_pos + t) * C);
  }
  parallel_for(T, [&](int t0, int t1) {
    for (int t = t0; t < t1; t++) {
      attention(config_, attn.data() + (size_t)t * C,
                qkv.data() + (size_t)t * 3 * C, K, V, start_pos + t);
    }
  });

  linear_q(out.data(), attn.data(), qparam(layer, "attn.c_proj.weight"),
           param(layer, "attn.c_proj.bias"), T, C, C);

  add_inplace(x, out);

  layernorm(normed.data(), x.data(), param(layer, "ln_2.weight"),
            param(layer, "ln_2.bias"), T, C);

  linear_q(hidden.data(), normed.data(), qparam(layer, "mlp.c_fc.weight"),
           param(layer, "mlp.c_fc.bias"), T, C, 4 * C);

  for (float &h : hidden)
    h = gelu(h);

  linear_q(out.data(), hidden.data(), qparam(layer, "mlp.c_proj.weight"),
           param(layer, "mlp.c_proj.bias"), T, 4 * C, C);

  add_inplace(x, out);
}
