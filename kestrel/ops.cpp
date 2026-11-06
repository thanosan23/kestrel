#include "ops.hpp"
#include "threadpool.hpp"
#include <algorithm>
#include <arm_neon.h>
#include <cmath>

// approximation from: https://arxiv.org/pdf/1606.08415
float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608f;
  return 0.5f * x *
         (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

QTensor quantize_rows(const std::vector<float> &w, int rows, int cols) {
  QTensor t;
  t.rows = rows;
  t.cols = cols;
  t.q.resize((size_t)rows * cols);
  t.scale.resize(rows);

  for (int r = 0; r < rows; r++) {
    const float *row = w.data() + (size_t)r * cols;
    float max_abs = 0.0f;
    for (int c = 0; c < cols; c++) {
      max_abs = std::max(max_abs, std::fabs(row[c]));
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    t.scale[r] = scale;
    for (int c = 0; c < cols; c++) {
      int v = (int)std::lround(row[c] / scale);
      t.q[(size_t)r * cols + c] = (int8_t)std::clamp(v, -127, 127);
    }
  }

  return t;
}

float dot(const float *a, const float *b, int n) {
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

float dot_q(const float *a, const int8_t *b, int n) {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);

  int i = 0;
  for (; i + 15 < n; i += 16) {
    int8x16_t q = vld1q_s8(b + i);
    int16x8_t lo = vmovl_s8(vget_low_s8(q));
    int16x8_t hi = vmovl_s8(vget_high_s8(q));
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i + 0), f0);
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), f1);
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), f2);
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), f3);
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  float sum = vaddvq_f32(acc);
  for (; i < n; i++)
    sum += a[i] * b[i];
  return sum;
}

void linear_q(float *out, const float *in, const QTensor &W,
                     const float *bias, int T, int in_dim, int out_dim) {
  constexpr int TB = 8;
  parallel_for(out_dim, [&](int j0, int j1) {
    for (int t0 = 0; t0 < T; t0 += TB) {
      const int t1 = std::min(T, t0 + TB);
      for (int t = t0; t < t1; t++) {
        for (int j = j0; j < j1; j++) {
          out[(size_t)t * out_dim + j] = bias[j];
        }
      }
      for (int k = 0; k < in_dim; k++) {
        const int8_t *q_row = W.q.data() + (size_t)k * out_dim;
        const float s = W.scale[k];
        for (int t = t0; t < t1; t++) {
          const float xs = in[(size_t)t * in_dim + k] * s;
          float *y = out + (size_t)t * out_dim;
          for (int j = j0; j < j1; j++) {
            y[j] += xs * q_row[j];
          }
        }
      }
    }
  });
}

void layernorm(float *out, const float *in, const float *gamma,
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

void add_inplace(std::vector<float> &x, const std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); i++) {
    x[i] += y[i];
  }
}
