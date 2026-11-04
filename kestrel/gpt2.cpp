#include "io.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
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

// approximation from: https://arxiv.org/pdf/1606.08415
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
      float dot = 0.0f;
      for (int i = 0; i < head_dim; i++) {
        dot += qh[i] * kh[i];
      }
      probs[t2] = dot * scale;
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

struct KVCache {
  std::vector<std::vector<float>> k, v;
  KVCache()
      : k(cfg::n_layer, std::vector<float>(cfg::n_ctx * cfg::C)),
        v(cfg::n_layer, std::vector<float>(cfg::n_ctx * cfg::C)) {}
};

class GPT2 {
public:
  explicit GPT2(std::map<std::string, Tensor> weights)
      : weights(std::move(weights)) {};

  std::vector<float> step(int token, int pos, KVCache &cache) const {
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
    for (int v = 0; v < V; v++) {
      const float *emb = wte + v * C;
      float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
      int c;
      for (c = 0; c + 3 < C; c += 4) {
        dot0 += normed[c + 0] * emb[c + 0];
        dot1 += normed[c + 1] * emb[c + 1];
        dot2 += normed[c + 2] * emb[c + 2];
        dot3 += normed[c + 3] * emb[c + 3];
      }
      for (; c < C; c++) {
        dot0 += normed[c] * emb[c];
      }
      logits[v] = dot0 + dot1 + dot2 + dot3;
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

  void transformer_block(int layer, std::vector<float> &x, int pos,
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
};

static int argmax(const std::vector<float> &v) {
  return std::max_element(v.begin(), v.end()) - v.begin();
}

static int sample(const std::vector<float> &logits, float temperature,
                  int top_k, std::mt19937 &rng) {
  if (temperature <= 0.0f)
    return argmax(logits);

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.size());
  for (int i = 0; i < (int)logits.size(); i++) {
    candidates.push_back({logits[i], i});
  }
  int k =
      (top_k > 0 && top_k < (int)candidates.size() ? top_k
                                                   : (int)candidates.size());
  std::partial_sort(
      candidates.begin(), candidates.begin() + k, candidates.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });
  candidates.resize(k);

  const float max_logit = candidates[0].first;
  std::vector<float> probs(k);
  for (int i = 0; i < k; i++) {
    probs[i] = std::exp((candidates[i].first - max_logit) / temperature);
  }
  std::discrete_distribution<int> pick(probs.begin(), probs.end());
  return candidates[pick(rng)].second;
}

static const char *usage = "Usage: ./bin/main [options]\n"
                           " -p TEXT    prompt\n"
                           " -n N       number of tokens to generate\n"
                           " --temp T   temperature\n"
                           " --top-k K  sample from K most likely tokens\n";

int main(int argc, char **argv) {
  using Clock = std::chrono::steady_clock;
  auto seconds_since = [](Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };

  std::string prompt = "Hello world";
  int n_new_tokens = 20;
  float temperature = 0.8f;
  int top_k = 40;
  unsigned seed = std::random_device{}();

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value after " << arg << '\n';
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "-p" || arg == "--prompt")
      prompt = value();
    else if (arg == "-n")
      n_new_tokens = std::stoi(value());
    else if (arg == "--temp")
      temperature = std::stof(value());
    else if (arg == "--top-k")
      top_k = std::stoi(value());
    else if (arg == "--seed") {
      seed = std::stoul(value());
    } else if (arg == "-h" || arg == "--help") {
      std::cout << usage;
      return 0;
    } else {
      std::cerr << "unknown option: " << arg << '\n';
      return 1;
    }
  }

  std::ifstream weight_file("./gpt2_weights.kestrel");
  if (!weight_file) {
    std::cerr << "could not open ./gpt2_weights.kestrel\n";
    return 1;
  }

  auto t0 = Clock::now();
  GPT2 model(IO::read_weights(weight_file));
  double load_s = seconds_since(t0);

  Tokenizer tok;
  if (!tok.load("./gpt2_tokenizer.kestrel")) {
    std::cerr << "could not load ./gpt2_tokenizer.kestrel\n";
    return 1;
  }

  std::vector<int> tokens = tok.encode(prompt);
  if (tokens.empty()) {
    std::cerr << "prompt is empty\n";
    return 1;
  }
  std::cout << prompt << std::flush;

  KVCache cache;
  auto t1 = Clock::now();
  std::vector<float> logits;
  for (int pos = 0; pos < (int)tokens.size(); pos++) {
    logits = model.step(tokens[pos], pos, cache);
  }
  double prompt_s = seconds_since(t1);

  std::mt19937 rng(seed);
  constexpr int END_OF_TEXT = 50256;
  auto t2 = Clock::now();
  int generated = 0;
  for (int i = 0; i < n_new_tokens; i++) {
    int next = sample(logits, temperature, top_k, rng);
    if (next == END_OF_TEXT)
      break;
    if ((int)tokens.size() >= cfg::n_ctx)
      break;
    tokens.push_back(next);
    std::cout << tok.decode({next}) << std::flush;
    logits = model.step(next, tokens.size() - 1, cache);
    generated++;
  }
  double gen_s = seconds_since(t2);

  double tok_per_s = generated / gen_s;

  std::cout << '\n';

  std::cerr << "\nload: " << load_s << "s\n"
            << "prompt: " << prompt_s << "s (" << tokens.size() - generated
            << " tokens)\n"
            << "generation: " << tok_per_s << " tokens/sec\n";

  return 0;
}
