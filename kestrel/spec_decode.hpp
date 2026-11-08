#pragma once

#include "gpt2.hpp"
#include "kv_cache.hpp"
#include <random>
#include <vector>
struct SpecStats {
  int rounds = 0;
  int drafted = 0;
  int accepted = 0;
};
std::vector<float> to_probs(const float *logits, float temperature, int top_k);

int sample(const std::vector<float> &probs, std::mt19937 &rng);

std::vector<int> spec_round(const GPT2 &draft, KVCache &dcache,
                            const GPT2 &target, KVCache &tcache, int last,
                            int k, float temperature, int top_k,
                            std::mt19937 &rng, SpecStats &stats);
