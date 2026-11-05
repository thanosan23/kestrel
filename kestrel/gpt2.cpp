#include "gpt2.hpp"
#include "config.hpp"
#include "threadpool.hpp"
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <stdexcept>

// approximation from: https://arxiv.org/pdf/1606.08415
static float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

static float dot(const float *a, const float *b, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  int i = 0;
  for (; i + 15 < n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i + 0), vld1q_f32(b + i + 0));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  float sum = vaddvq_f32(acc);
  for (; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}

static void linear(float *out, const float *in, const float *W,
                   const float *bias, int T, int in_dim, int out_dim) {
  for (int t = 0; t < T; t++) {
    const float *x = in + t * in_dim;
    float *y = out + t * out_dim;

    parallel_for(out_dim, [&](int j0, int j1) {
      for (int j = j0; j < j1; j++)
        y[j] = bias[j];

      for (int k = 0; k < in_dim; k++) {
        const float *W_row = W + k * out_dim;
        for (int j = j0; j < j1; j++)
          y[j] += x[k] * W_row[j];
      }
    });
  }
}

static void layernorm(float *out, const float *in, const float *gamma,
                      const float *beta, int T, int C) {
  const float eps = 1e-5f;
  for (int t = 0; t < T; t++) {
    const float *x = in + t * C;
    float *y = out + t * C;

    float mean = 0.0f;
    for (int i = 0; i < C; i++)
      mean += x[i];
    mean /= C;

    float var = 0.0f;
    for (int i = 0; i < C; i++)
      var += (x[i] - mean) * (x[i] - mean);
    var /= C;

    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < C; i++)
      y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
  }
}

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

static void add_inplace(std::vector<float> &x, const std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); i++) {
    x[i] += y[i];
  }
}

std::vector<float> GPT2::step(int token, int pos, KVCache &cache) const {
  using namespace cfg;

  if (pos >= n_ctx) {
    throw std::runtime_error("context is full");
  }

  const float *wte = param("wte.weight");
  const float *wpe = param("wpe.weight");
  std::vector<float> x(C);
  for (int c = 0; c < C; c++) {
    x[c] = wte[token * C + c] + wpe[pos * C + c];
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
      logits[v] = dot(normed.data(), wte + v * C, C);
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

  linear(qkv.data(), normed.data(), param(layer, "attn.c_attn.weight"),
         param(layer, "attn.c_attn.bias"), 1, C, 3 * C);

  float *K = cache.k[layer].data();
  float *V = cache.v[layer].data();

  std::copy(qkv.begin() + C, qkv.begin() + 2 * C, K + pos * C);
  std::copy(qkv.begin() + 2 * C, qkv.end(), V + pos * C);

  attention(attn.data(), qkv.data(), K, V, pos);

  linear(out.data(), attn.data(), param(layer, "attn.c_proj.weight"),
         param(layer, "attn.c_proj.bias"), 1, C, C);

  add_inplace(x, out);

  layernorm(normed.data(), x.data(), param(layer, "ln_2.weight"),
            param(layer, "ln_2.bias"), 1, C);

  linear(hidden.data(), normed.data(), param(layer, "mlp.c_fc.weight"),
         param(layer, "mlp.c_fc.bias"), 1, C, 4 * C);

  for (float &h : hidden)
    h = gelu(h);

  linear(out.data(), hidden.data(), param(layer, "mlp.c_proj.weight"),
         param(layer, "mlp.c_proj.bias"), 1, 4 * C, C);

  add_inplace(x, out);
}
