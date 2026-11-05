#pragma once

namespace cfg {
constexpr int n_layer = 12;
constexpr int n_head = 12;
constexpr int C = 768;
constexpr int head_dim = C / n_head;
constexpr int n_ctx = 1024;
constexpr int V = 50257;
} // namespace cfg
