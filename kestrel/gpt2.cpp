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

static void attention(float *out, const float *q, const float *K,
                      const float *V, int pos) {
  using namespace cfg;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  std::vector<float> probs(pos + 1);

  for (int h = 0; h < n_head; h++) {
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

GPT2::GPT2(std::map<std::string, Tensor> w) : weights(std::move(w)) {
  using namespace cfg;
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
  quantize("wte.weight", V, C);
}

std::vector<float> GPT2::prefill(const std::vector<int> &tokens,
                                 KVCache &cache) const {
  using namespace cfg;
  const int T = tokens.size();
  if (T == 0 || T > n_ctx) {
    throw std::runtime_error("prompt must be between 1 to 1024 tokens");
  }
  const QTensor &wte = qparam("wte.weight");
  const float *wpe = param("wpe.weight");

  std::vector<float> x((size_t)T * C);
  for (int t = 0; t < T; t++) {
    const int8_t *tok_row = wte.q.data() + tokens[t] * C;
    const float s = wte.scale[tokens[t]];
    for (int c = 0; c < C; c++) {
      x[(size_t)t * C + c] = tok_row[c] * s + wpe[t * C + c];
    }
  }
  for (int layer = 0; layer < n_layer; layer++) {
    transformer_block_batched(layer, x, T, cache);
  }
  std::vector<float> normed(C);
  layernorm(normed.data(), x.data() + (size_t)(T - 1) * C, param("ln_f.weight"),
            param("ln_f.bias"), 1, C);

  std::vector<float> logits(V);
  parallel_for(V, [&](int v0, int v1) {
    for (int v = v0; v < v1; v++) {
      logits[v] =
          wte.scale[v] * dot_q(normed.data(), wte.q.data() + (size_t)v * C, C);
    }
  });
  return logits;
}

void GPT2::transformer_block_batched(int layer, std::vector<float> &x, int T,
                                     KVCache &cache) const {
  using namespace cfg;
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
    std::copy(row + C, row + 2 * C, K + (size_t)t * C);
    std::copy(row + 2 * C, row + 3 * C, V + (size_t)t * C);
  }
  parallel_for(T, [&](int t0, int t1) {
    for (int t = t0; t < t1; t++) {
      attention(attn.data() + (size_t)t * C, qkv.data() + (size_t)t * 3 * C, K,
                V, t);
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

std::vector<float> GPT2::step(int token, int pos, KVCache &cache) const {
  using namespace cfg;

  if (pos >= n_ctx) {
    throw std::runtime_error("context is full");
  }

  const QTensor &wte = qparam("wte.weight");
  const float *wpe = param("wpe.weight");
  std::vector<float> x(C);
  const int8_t *tok_row = wte.q.data() + (size_t)token * C;
  for (int c = 0; c < C; c++) {
    x[c] = tok_row[c] * wte.scale[token] + wpe[pos * C + c];
  }
  for (int layer = 0; layer < n_layer; layer++) {
    transformer_block(layer, x, pos, cache);
  }

  std::vector<float> normed(C);
  layernorm(normed.data(), x.data(), param("ln_f.weight"), param("ln_f.bias"),
            1, C);

  std::vector<float> logits(V);
  parallel_for(V, [&](int v0, int v1) {
    for (int v = v0; v < v1; v++) {
      logits[v] =
          wte.scale[v] * dot_q(normed.data(), wte.q.data() + (size_t)v * C, C);
    }
  });
  return logits;
}

void GPT2::transformer_block(int layer, std::vector<float> &x, int pos,
                             KVCache &cache) const {
  using namespace cfg;

  std::vector<float> normed(C), qkv(3 * C), attn(C), hidden(4 * C), out(C);

  layernorm(normed.data(), x.data(), param(layer, "ln_1.weight"),
            param(layer, "ln_1.bias"), 1, C);

  linear_q(qkv.data(), normed.data(), qparam(layer, "attn.c_attn.weight"),
           param(layer, "attn.c_attn.bias"), 1, C, 3 * C);

  float *K = cache.k[layer].data();
  float *V = cache.v[layer].data();

  std::copy(qkv.begin() + C, qkv.begin() + 2 * C, K + pos * C);
  std::copy(qkv.begin() + 2 * C, qkv.end(), V + pos * C);

  attention(attn.data(), qkv.data(), K, V, pos);

  linear_q(out.data(), attn.data(), qparam(layer, "attn.c_proj.weight"),
           param(layer, "attn.c_proj.bias"), 1, C, C);

  add_inplace(x, out);

  layernorm(normed.data(), x.data(), param(layer, "ln_2.weight"),
            param(layer, "ln_2.bias"), 1, C);

  linear_q(hidden.data(), normed.data(), qparam(layer, "mlp.c_fc.weight"),
           param(layer, "mlp.c_fc.bias"), 1, C, 4 * C);

  for (float &h : hidden)
    h = gelu(h);

  linear_q(out.data(), hidden.data(), qparam(layer, "mlp.c_proj.weight"),
           param(layer, "mlp.c_proj.bias"), 1, 4 * C, C);

  add_inplace(x, out);
}
