#pragma once

#include "tensor.hpp"
#include <cstdint>
#include <vector>

float gelu(float x);
QTensor quantize_rows(const std::vector<float> &w, int rows, int cols);
float dot(const float *a, const float *b, int n);
float dot_q(const float *a, const int8_t *b, int n);
void linear_q(float *out, const float *in, const QTensor &W, const float *bias,
              int T, int in_dim, int out_dim);
void layernorm(float *out, const float *in, const float *gamma,
               const float *beta, int T, int C);
void add_inplace(std::vector<float> &x, const std::vector<float> &y);
