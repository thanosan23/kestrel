#pragma once

struct GPT2Config {
  int n_layer, n_head, C;
  static constexpr int head_dim = 64;
  static constexpr int n_ctx = 1024;
  static constexpr int V = 50257;
}; // namespace GPT2Config

namespace gpt {
inline constexpr GPT2Config small{12, 12, 768};
inline constexpr GPT2Config medium{24, 16, 1024};
inline constexpr GPT2Config large{36, 20, 1280};
inline constexpr GPT2Config xl{48, 25, 1600};
}; // namespace gpt
