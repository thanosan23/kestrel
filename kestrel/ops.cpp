#include "ops.hpp"
#include "tensor.hpp"
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

template <int NT>
static void dot_q_rows(const float *x, int x_stride, const int8_t *w, int n,
                       float *result) {
  float32x4_t acc[NT][4];
  for (int t = 0; t < NT; t++)
    for (int c = 0; c < 4; c++)
      acc[t][c] = vdupq_n_f32(0.0f);

  int i = 0;
  for (; i + 15 < n; i += 16) {
    int8x16_t q = vld1q_s8(w + i);
    int16x8_t lo = vmovl_s8(vget_low_s8(q));
    int16x8_t hi = vmovl_s8(vget_high_s8(q));
    float32x4_t f[4] = {
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
    };
    for (int t = 0; t < NT; t++) {
      const float *xt = x + (size_t)t * x_stride + i;
      for (int c = 0; c < 4; c++)
        acc[t][c] = vfmaq_f32(acc[t][c], vld1q_f32(xt + 4 * c), f[c]);
    }
  }
  for (int t = 0; t < NT; t++) {
    float32x4_t a = vaddq_f32(vaddq_f32(acc[t][0], acc[t][1]),
                              vaddq_f32(acc[t][2], acc[t][3]));
    float sum = vaddvq_f32(a);
    const float *xt = x + (size_t)t * x_stride;
    for (int j = i; j < n; j++)
      sum += xt[j] * w[j];
    result[t] = sum;
  }
}

constexpr int QBLOCK = 32;

#if defined(__ARM_FEATURE_DOTPROD)
template <int NT>
static void dot_q8_rows(const int8_t *x, const float *xs, int x_stride,
                        const int8_t *w, int n, float *result) {
  const int nb = n / QBLOCK;
  const int s_stride = x_stride / QBLOCK;
  float32x4_t acc[NT];
  for (int t = 0; t < NT; t++)
    acc[t] = vdupq_n_f32(0.0f);

  for (int b = 0; b < nb; b++) {
    const int8x16_t w0 = vld1q_s8(w + b * QBLOCK);
    const int8x16_t w1 = vld1q_s8(w + b * QBLOCK + 16);
    for (int t = 0; t < NT; t++) {
      const int8_t *xt = x + (size_t)t * x_stride + b * QBLOCK;
      int32x4_t d = vdotq_s32(vdupq_n_s32(0), w0, vld1q_s8(xt));
      d = vdotq_s32(d, w1, vld1q_s8(xt + 16));
      acc[t] =
          vfmaq_n_f32(acc[t], vcvtq_f32_s32(d), xs[(size_t)t * s_stride + b]);
    }
  }
  for (int t = 0; t < NT; t++)
    result[t] = vaddvq_f32(acc[t]);
}

static void quantize_activations(const float *in, int T, int n,
                                 std::vector<int8_t> &q,
                                 std::vector<float> &scale) {
  const int nb = n / QBLOCK;
  q.resize((size_t)T * n);
  scale.resize((size_t)T * nb);
  for (int t = 0; t < T; t++) {
    for (int b = 0; b < nb; b++) {
      const float *x = in + (size_t)t * n + b * QBLOCK;
      int8_t *qx = q.data() + (size_t)t * n + b * QBLOCK;
      float max_abs = 0.0f;
      for (int i = 0; i < QBLOCK; i++)
        max_abs = std::max(max_abs, std::fabs(x[i]));
      const float s = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
      const float inv = 1.0f / s;
      for (int i = 0; i < QBLOCK; i++)
        qx[i] = (int8_t)std::nearbyint(x[i] * inv);
      scale[(size_t)t * nb + b] = s;
    }
  }
}
#endif
static void dot_q_group(int nt, const float *x, int stride, const int8_t *w,
                        int n, float *res) {
  switch (nt) {
  case 4:
    dot_q_rows<4>(x, stride, w, n, res);
    break;
  case 3:
    dot_q_rows<3>(x, stride, w, n, res);
    break;
  case 2:
    dot_q_rows<2>(x, stride, w, n, res);
    break;
  default:
    dot_q_rows<1>(x, stride, w, n, res);
    break;
  }
}

#if defined(__ARM_FEATURE_DOTPROD)
static void dot_q8_group(int nt, const int8_t *x, const float *xs, int stride,
                         const int8_t *w, int n, float *res) {
  switch (nt) {
  case 4:
    dot_q8_rows<4>(x, xs, stride, w, n, res);
    break;
  case 3:
    dot_q8_rows<3>(x, xs, stride, w, n, res);
    break;
  case 2:
    dot_q8_rows<2>(x, xs, stride, w, n, res);
    break;
  default:
    dot_q8_rows<1>(x, xs, stride, w, n, res);
    break;
  }
}
#endif

void linear_q(float *out, const float *in, const QTensor &W, const float *bias,
              int T, int in_dim, int out_dim) {
  constexpr int TB = 32;

#if defined(__ARM_FEATURE_DOTPROD)
  const bool use_sdot = in_dim % QBLOCK == 0;
  static thread_local std::vector<int8_t> qx;
  static thread_local std::vector<float> sx;
  if (use_sdot)
    quantize_activations(in, T, in_dim, qx, sx);
  const int nb = in_dim / QBLOCK;
  const int8_t *qx_data = qx.data();
  const float *sx_data = sx.data();
#endif

  parallel_for(out_dim, [&](int r0, int r1) {
    for (int t0 = 0; t0 < T; t0 += TB) {
      const int t1 = std::min(T, t0 + TB);
      for (int r = r0; r < r1; r++) {
        const int8_t *w = W.q.data() + (size_t)r * in_dim;
        const float s = W.scale[r];
        const float b = bias ? bias[r] : 0.0f;
        for (int t = t0; t < t1; t += 4) {
          const int nt = std::min(4, t1 - t);
          float res[4];
#if defined(__ARM_FEATURE_DOTPROD)
          if (use_sdot)
            dot_q8_group(nt, qx_data + (size_t)t * in_dim,
                         sx_data + (size_t)t * nb, in_dim, w, in_dim, res);
          else
#endif
            dot_q_group(nt, in + (size_t)t * in_dim, in_dim, w, in_dim, res);
          for (int u = 0; u < nt; u++)
            out[(size_t)(t + u) * out_dim + r] = s * res[u] + b;
        }
      }
    }
  });
}

QTensor quantize_transposed(const std::vector<float> &w, int in_dim,
                            int out_dim) {
  std::vector<float> t((size_t)in_dim * out_dim);
  for (int i = 0; i < in_dim; i++) {
    for (int o = 0; o < out_dim; o++) {
      t[(size_t)o * in_dim + i] = w[(size_t)i * out_dim + o];
    }
  }
  return quantize_rows(t, out_dim, in_dim);
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
