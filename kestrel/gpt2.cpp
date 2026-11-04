#include "io.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cfg {
constexpr int n_layer = 12;
constexpr int n_head = 12;
constexpr int C = 768;
constexpr int head_dim = C / n_head;
constexpr int n_ctx = 1024;
constexpr int V = 50257;
} // namespace cfg

static float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

static void linear(float *out, const float *in, const float *W,
                   const float *bias, int T, int in_dim, int out_dim) {
  for (int t = 0; t < T; t++) {
    const float *x = in + t * in_dim;
    float *y = out + t * out_dim;

    for (int j = 0; j < out_dim; j++)
      y[j] = bias[j];

    for (int k = 0; k < in_dim; k++) {
      const float *W_row = W + k * out_dim;
      for (int j = 0; j < out_dim; j++)
        y[j] += x[k] * W_row[j];
    }
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

static void causal_self_attention(float *out, const float *qkv, int T) {
  using namespace cfg;
  const float scale = 1.0 / std::sqrt((float)head_dim);

  auto query = [&](int t, int h) {
    return qkv + t * 3 * C + 0 * C + h * head_dim;
  };

  auto key = [&](int t, int h) {
    return qkv + t * 3 * C + 1 * C + h * head_dim;
  };

  auto value = [&](int t, int h) {
    return qkv + t * 3 * C + 2 * C + h * head_dim;
  };

  std::vector<float> probs(T);
  for (int h = 0; h < n_head; h++) {
    for (int t = 0; t < T; t++) {
      const float *q = query(t, h);

      float max_score = -INFINITY;
      for (int t2 = 0; t2 <= t; t2++) {
        const float *k = key(t2, h);
        float dot = 0.0f;
        for (int i = 0; i < head_dim; i++)
          dot += q[i] * k[i];
        probs[t2] = dot * scale;
        max_score = std::max(max_score, probs[t2]);
      }

      float sum = 0.0f;
      for (int t2 = 0; t2 <= t; t2++) {
        probs[t2] = std::exp(probs[t2] - max_score);
        sum += probs[t2];
      }
      for (int t2 = 0; t2 <= t; t2++)
        probs[t2] /= sum;

      float *o = out + t * C + h * head_dim;
      std::fill(o, o + head_dim, 0.0f);
      for (int t2 = 0; t2 <= t; t2++) {
        const float *v = value(t2, h);
        for (int i = 0; i < head_dim; i++)
          o[i] += probs[t2] * v[i];
      }
    }
  }
}

static void add_inplace(std::vector<float> &x, const std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); i++) {
    x[i] += y[i];
  }
}

class GPT2 {
public:
  explicit GPT2(std::map<std::string, Tensor> weights)
      : weights(std::move(weights)) {};

  std::vector<float> next_token_logits(const std::vector<int> &tokens) const {
    using namespace cfg;

    const int T = tokens.size();
    if (T == 0 || T > n_ctx) {
      throw std::runtime_error("sequence length must be between 1 and 1024");
    }

    std::vector<float> x(T * C);
    const float *wte = param("wte.weight");
    const float *wpe = param("wpe.weight");
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < C; c++) {
        x[t * C + c] = wte[tokens[t] * C + c] + wpe[t * C + c];
      }
    }
    for (int layer = 0; layer < n_layer; layer++) {
      transformer_block(layer, x, T);
    }

    std::vector<float> normed(T * C);
    layernorm(normed.data(), x.data(), param("ln_f.weight"), param("ln_f.bias"),
              T, C);

    const float *last = normed.data() + (T - 1) * C;
    std::vector<float> logits(V);
    for (int v = 0; v < V; v++) {
      const float *emb = wte + v * C;
      float dot = 0.0f;
      for (int c = 0; c < C; c++)
        dot += last[c] * emb[c];
      logits[v] = dot;
    }
    return logits;
  }

private:
  std::map<std::string, Tensor> weights;

  const float *param(const std::string &name) const {
    return weights.at(name).data.data();
  }

  const float *param(int layer, const std::string &name) const {
    return param("h." + std::to_string(layer) + "." + name);
  }

  void transformer_block(int layer, std::vector<float> &x, int T) const {
    using namespace cfg;

    std::vector<float> normed(T * C), qkv(T * 3 * C), attn(T * C),
        hidden(T * 4 * C), out(T * C);

    layernorm(normed.data(), x.data(), param(layer, "ln_1.weight"),
              param(layer, "ln_1.bias"), T, C);

    linear(qkv.data(), normed.data(), param(layer, "attn.c_attn.weight"),
           param(layer, "attn.c_attn.bias"), T, C,
           3 * C); // [T, C] -> [T, 3*C]

    causal_self_attention(attn.data(), qkv.data(), T); // [T, 3C] --> [T, C]

    linear(out.data(), attn.data(), param(layer, "attn.c_proj.weight"),
           param(layer, "attn.c_proj.bias"), T, C, C);

    add_inplace(x, out);

    layernorm(normed.data(), x.data(), param(layer, "ln_2.weight"),
              param(layer, "ln_2.bias"), T, C);

    linear(hidden.data(), normed.data(), param(layer, "mlp.c_fc.weight"),
           param(layer, "mlp.c_fc.bias"), T, C, 4 * C);

    for (float &h : hidden)
      h = gelu(h);

    linear(out.data(), hidden.data(), param(layer, "mlp.c_proj.weight"),
           param(layer, "mlp.c_proj.bias"), T, 4 * C, C);

    add_inplace(x, out);
  }
};

static int argmax(const std::vector<float> &v) {
  return std::max_element(v.begin(), v.end()) - v.begin();
}

int main() {
  std::ifstream weight_file("./gpt2_weights.kestrel");
  if (!weight_file) {
    std::cerr << "could not open ./gpt2_weights.kestrel\n";
    return 1;
  }

  GPT2 model(IO::read_weights(weight_file));

  Tokenizer tok;
  if (!tok.load("./gpt2_tokenizer.kestrel")) {
    std::cerr << "could not load ../gpt2_tokenizer.kestrel\n";
    return 1;
  }

  const std::string prompt = "Hello world";
  int n_new_tokens = 20;
  std::vector<int> tokens = tok.encode(prompt);
  std::cout << prompt << std::flush;

  for (int i = 0; i < n_new_tokens; i++) {
    int next = argmax(model.next_token_logits(tokens));
    tokens.push_back(next);
    std::cout << tok.decode({next}) << std::flush;
  }
  std::cout << '\n';
  return 0;
}
