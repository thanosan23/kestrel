#include "spec_decode.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

std::vector<float> to_probs(const float *logits, float temperature, int top_k) {
  constexpr int V = GPT2Config::V;

  std::vector<float> probs(V, 0.0f);

  if (temperature <= 0.0f) {
    probs[std::max_element(logits, logits + V) - logits] = 1.0f;
    return probs;
  }

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(V);
  for (int i = 0; i < V; i++) {
    candidates.push_back({logits[i], i});
  }

  int k = (top_k > 0 && top_k < V) ? top_k : V;
  std::partial_sort(
      candidates.begin(), candidates.begin() + k, candidates.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });
  const float max_logit = candidates[0].first;

  float sum = 0.0f;
  for (int i = 0; i < k; i++) {
    float p = std::exp((candidates[i].first - max_logit) / temperature);
    probs[candidates[i].second] = p;
    sum += p;
  }
  for (int i = 0; i < k; i++) {
    probs[candidates[i].second] /= sum;
  }
  return probs;
}

int sample(const std::vector<float> &probs, std::mt19937 &rng) {
  std::discrete_distribution<int> pick(probs.begin(), probs.end());
  return pick(rng);
}

std::vector<int> spec_round(const GPT2 &draft, KVCache &dcache,
                            const GPT2 &target, KVCache &tcache, int last,
                            int k, float temperature, int top_k,
                            std::mt19937 &rng, SpecStats &stats) {
  const int V = GPT2Config::V;
  const int L = tcache.len;

  std::vector<int> drafted;
  std::vector<std::vector<float>> q;
  int cur = last;
  for (int i = 0; i < k; i++) {
    std::vector<float> lg = draft.forward({cur}, dcache);
    q.push_back(to_probs(lg.data(), temperature, top_k));
    cur = sample(q.back(), rng);
    drafted.push_back(cur);
  }
  std::vector<int> verify = {last};
  verify.insert(verify.end(), drafted.begin(), drafted.end());
  std::vector<float> vlogits = target.forward(verify, tcache, true);

  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<int> out;

  int n = 0;
  int next = -1;
  for (; n < k; n++) {
    std::vector<float> p =
        to_probs(vlogits.data() + (size_t)n * V, temperature, top_k);
    const int x = drafted[n];
    if (uniform(rng) < p[x] / q[n][x]) {
      out.push_back(x);
      continue;
    }

    std::vector<float> residual(V);
    float sum = 0.0f;
    for (int v = 0; v < V; v++) {
      residual[v] = std::max(0.0f, p[v] - q[n][v]);
      sum += residual[v];
    }
    next = sum > 0.0f ? sample(residual, rng) : sample(p, rng);
    break;
  }
  if (n == k) {
    next = sample(to_probs(vlogits.data() + (size_t)k * V, temperature, top_k),
                  rng);
  }
  out.push_back(next);

  stats.rounds++;
  stats.drafted += k;
  stats.accepted += n;

  tcache.truncate(L + 1 + n);
  if (n == k)
    draft.forward({drafted.back()}, dcache);
  else
    dcache.truncate(L + 1 + n);
  return out;
}
