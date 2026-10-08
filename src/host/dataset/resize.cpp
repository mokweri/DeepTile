/**
 * @file resize.cpp
 * @brief Implements Pillow-compatible antialiased bilinear resizing.
 *
 * This is a port of Pillow's `Resample.c` for the bilinear filter. The
 * fixed-point coefficient format, the pass order, and the intermediate
 * clamping are all part of the contract: changing any of them breaks bit-exact
 * agreement with the exporter.
 *
 * Like Pillow, the passes accumulate in 32 bits. On AArch64 the vertical
 * pass runs on NEON; the integer arithmetic, and therefore the output, is
 * identical to the scalar path.
 */

#include "dataset/resize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define DEEPTILE_RESIZE_NEON 1
#endif

namespace deeptile {
namespace dataset {
namespace {

// Pillow accumulates in 32-bit with 8 bits of sample and 2 bits of headroom.
constexpr int kPrecisionBits = 32 - 8 - 2;

// Bilinear support radius before downscale stretching.
constexpr double kBilinearSupport = 1.0;

// Half a fixed-point unit, added before the shift so it rounds rather than
// truncates.
constexpr std::int32_t kRoundingBias = std::int32_t{1} << (kPrecisionBits - 1);

double triangle_filter(double x) {
  const double a = std::abs(x);
  return a < 1.0 ? 1.0 - a : 0.0;
}

/** Per-output-index filter taps for one axis. */
struct AxisCoeffs {
  int ksize = 0;
  std::vector<int> offset;  // First contributing source index.
  std::vector<int> length;  // Number of contributing source samples.
  std::vector<std::int32_t> taps;  // out_size * ksize, fixed point.
};

/**
 * Port of Pillow's precompute_coeffs().
 *
 * The filter is stretched by the downscale ratio, which is what produces
 * antialiasing; for upscaling the ratio clamps to 1 and the filter degenerates
 * to plain bilinear interpolation.
 */
AxisCoeffs precompute_coeffs(int in_size, int out_size) {
  const double scale =
      static_cast<double>(in_size) / static_cast<double>(out_size);
  const double filterscale = std::max(scale, 1.0);
  const double support = kBilinearSupport * filterscale;

  AxisCoeffs c;
  c.ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
  c.offset.resize(static_cast<std::size_t>(out_size));
  c.length.resize(static_cast<std::size_t>(out_size));
  c.taps.assign(static_cast<std::size_t>(out_size) * c.ksize, 0);

  std::vector<double> weights(static_cast<std::size_t>(c.ksize));
  for (int i = 0; i < out_size; ++i) {
    const double center = (static_cast<double>(i) + 0.5) * scale;
    const double inv_filterscale = 1.0 / filterscale;

    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0) {
      xmin = 0;
    }
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size) {
      xmax = in_size;
    }
    xmax -= xmin;

    std::fill(weights.begin(), weights.end(), 0.0);
    double total = 0.0;
    for (int x = 0; x < xmax; ++x) {
      const double w = triangle_filter(
          (static_cast<double>(x + xmin) - center + 0.5) * inv_filterscale);
      weights[static_cast<std::size_t>(x)] = w;
      total += w;
    }
    if (total != 0.0) {
      for (int x = 0; x < xmax; ++x) {
        weights[static_cast<std::size_t>(x)] /= total;
      }
    }

    c.offset[static_cast<std::size_t>(i)] = xmin;
    c.length[static_cast<std::size_t>(i)] = xmax;
    // Pillow rounds coefficients away from zero before truncating to integer.
    for (int x = 0; x < c.ksize; ++x) {
      const double w = weights[static_cast<std::size_t>(x)];
      const double scaled = w * static_cast<double>(1 << kPrecisionBits);
      c.taps[static_cast<std::size_t>(i) * c.ksize + x] =
          static_cast<std::int32_t>(w < 0.0 ? scaled - 0.5 : scaled + 0.5);
    }
  }
  return c;
}

/**
 * True when every output's accumulation stays within 32 bits. The triangle
 * filter's taps are non-negative and sum to about 2^22, so 255 times their sum
 * plus the rounding bias is near 2^30; this check makes that a guarantee
 * rather than an assumption, and anything that fails it takes the 64-bit path.
 */
bool fits_int32(const AxisCoeffs &c) {
  const std::size_t outputs = c.offset.size();
  for (std::size_t i = 0; i < outputs; ++i) {
    std::int64_t positive = 0;
    const std::int32_t *taps = c.taps.data() + i * c.ksize;
    for (int t = 0; t < c.length[i]; ++t) {
      if (taps[t] < 0) {
        return false;
      }
      positive += taps[t];
    }
    if (255 * positive + kRoundingBias > INT32_MAX) {
      return false;
    }
  }
  return true;
}

std::uint8_t clip8(std::int64_t accumulator) {
  const std::int64_t v = accumulator >> kPrecisionBits;
  if (v < 0) {
    return 0;
  }
  if (v > 255) {
    return 255;
  }
  return static_cast<std::uint8_t>(v);
}

std::uint8_t clip8_32(std::int32_t accumulator) {
  const std::int32_t v = accumulator >> kPrecisionBits;
  return static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
}

Image8 make_image(int width, int height, int channels) {
  Image8 dst;
  dst.width = width;
  dst.height = height;
  dst.channels = channels;
  dst.pixels.resize(dst.expected_size());
  return dst;
}

/**
 * Horizontal pass producing output columns [x0, x0 + width) for source rows
 * [y0, y0 + height). Coefficients are indexed by absolute output column, so a
 * window yields exactly the same bytes as the full pass.
 */
Image8 resample_horizontal(const Image8 &src, int src_x, int src_y,
                           const AxisCoeffs &c, int x0, int width, int y0,
                           int height) {
  Image8 dst = make_image(width, height, src.channels);
  const int ch = src.channels;
  const bool narrow = fits_int32(c);

  for (int y = 0; y < height; ++y) {
    // Coefficient offsets are absolute; the region starts at (src_x, src_y).
    const std::uint8_t *row =
        src.pixels.data() + static_cast<std::size_t>(y0 + y - src_y) * src.width * ch;
    std::uint8_t *out_row =
        dst.pixels.data() + static_cast<std::size_t>(y) * width * ch;
    for (int x = 0; x < width; ++x) {
      const std::size_t xi = static_cast<std::size_t>(x0 + x);
      const int len = c.length[xi];
      const std::int32_t *taps = c.taps.data() + xi * c.ksize;
      const std::uint8_t *in =
          row + static_cast<std::size_t>(c.offset[xi] - src_x) * ch;
      std::uint8_t *out = out_row + static_cast<std::size_t>(x) * ch;
      if (narrow && ch == 3) {
        // Scalar on every target: per-pixel NEON here loses to the scalar
        // loop on the Cortex-A53, because each tap moves data between the
        // general and vector registers.
        std::int32_t a0 = kRoundingBias;
        std::int32_t a1 = kRoundingBias;
        std::int32_t a2 = kRoundingBias;
        for (int t = 0; t < len; ++t) {
          const std::int32_t k = taps[t];
          a0 += in[t * 3 + 0] * k;
          a1 += in[t * 3 + 1] * k;
          a2 += in[t * 3 + 2] * k;
        }
        out[0] = clip8_32(a0);
        out[1] = clip8_32(a1);
        out[2] = clip8_32(a2);
      } else if (narrow) {
        for (int k = 0; k < ch; ++k) {
          std::int32_t acc = kRoundingBias;
          for (int t = 0; t < len; ++t) {
            acc += in[t * ch + k] * taps[t];
          }
          out[k] = clip8_32(acc);
        }
      } else {
        for (int k = 0; k < ch; ++k) {
          std::int64_t acc = kRoundingBias;
          for (int t = 0; t < len; ++t) {
            acc += static_cast<std::int64_t>(in[t * ch + k]) * taps[t];
          }
          out[k] = clip8(acc);
        }
      }
    }
  }
  return dst;
}

/**
 * One output row of the vertical pass: every byte of the row combines the same
 * source rows with the same taps, so the row is processed as a flat vector.
 */
void vertical_row(const std::uint8_t *base, std::size_t stride, int len,
                  const std::int32_t *taps, bool narrow, std::uint8_t *out) {
  std::size_t i = 0;
#ifdef DEEPTILE_RESIZE_NEON
  if (narrow) {
    for (; i + 16 <= stride; i += 16) {
      int32x4_t a0 = vdupq_n_s32(kRoundingBias);
      int32x4_t a1 = a0;
      int32x4_t a2 = a0;
      int32x4_t a3 = a0;
      for (int t = 0; t < len; ++t) {
        const uint8x16_t v = vld1q_u8(base + static_cast<std::size_t>(t) * stride + i);
        const uint16x8_t lo = vmovl_u8(vget_low_u8(v));
        const uint16x8_t hi = vmovl_u8(vget_high_u8(v));
        const std::int32_t k = taps[t];
        a0 = vmlaq_n_s32(a0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))), k);
        a1 = vmlaq_n_s32(a1, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo))), k);
        a2 = vmlaq_n_s32(a2, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))), k);
        a3 = vmlaq_n_s32(a3, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi))), k);
      }
      const uint16x8_t n0 = vcombine_u16(vqmovun_s32(vshrq_n_s32(a0, kPrecisionBits)),
                                         vqmovun_s32(vshrq_n_s32(a1, kPrecisionBits)));
      const uint16x8_t n1 = vcombine_u16(vqmovun_s32(vshrq_n_s32(a2, kPrecisionBits)),
                                         vqmovun_s32(vshrq_n_s32(a3, kPrecisionBits)));
      vst1q_u8(out + i, vcombine_u8(vqmovn_u16(n0), vqmovn_u16(n1)));
    }
  }
#endif
  for (; i < stride; ++i) {
    if (narrow) {
      std::int32_t acc = kRoundingBias;
      for (int t = 0; t < len; ++t) {
        acc += base[static_cast<std::size_t>(t) * stride + i] * taps[t];
      }
      out[i] = clip8_32(acc);
    } else {
      std::int64_t acc = kRoundingBias;
      for (int t = 0; t < len; ++t) {
        acc += static_cast<std::int64_t>(base[static_cast<std::size_t>(t) * stride + i]) *
               taps[t];
      }
      out[i] = clip8(acc);
    }
  }
}

/**
 * Vertical pass producing output rows [y0, y0 + height). `src` holds the rows
 * of the horizontal pass starting at absolute row `src_row0`.
 */
Image8 resample_vertical(const Image8 &src, const AxisCoeffs &c, int y0,
                         int height, int src_row0) {
  Image8 dst = make_image(src.width, height, src.channels);
  const std::size_t stride = static_cast<std::size_t>(src.width) * src.channels;
  const bool narrow = fits_int32(c);
  for (int y = 0; y < height; ++y) {
    const std::size_t yi = static_cast<std::size_t>(y0 + y);
    const int ymin = c.offset[yi] - src_row0;
    vertical_row(src.pixels.data() + static_cast<std::size_t>(ymin) * stride,
                 stride, c.length[yi], c.taps.data() + yi * c.ksize, narrow,
                 dst.pixels.data() + static_cast<std::size_t>(y) * stride);
  }
  return dst;
}

/** Source span [first, first + count) that outputs [y0, y0 + height) read. */
void source_span(const AxisCoeffs &c, int in_size, int y0, int height,
                 int &first_row, int &row_count) {
  int lo = in_size;
  int hi = 0;
  for (int y = y0; y < y0 + height; ++y) {
    const std::size_t yi = static_cast<std::size_t>(y);
    lo = std::min(lo, c.offset[yi]);
    hi = std::max(hi, c.offset[yi] + c.length[yi]);
  }
  first_row = lo;
  row_count = hi - lo;
}

} // namespace

namespace {

bool valid_window(int out_width, int out_height, int window_x, int window_y,
                  int window_width, int window_height) {
  return out_width > 0 && out_height > 0 && window_width > 0 &&
         window_height > 0 && window_x >= 0 && window_y >= 0 &&
         window_x + window_width <= out_width &&
         window_y + window_height <= out_height;
}

/** Source span along one axis read by outputs [w0, w0 + wn). */
void axis_span(int full, int out, int w0, int wn, int &lo, int &hi) {
  if (out == full) {
    lo = w0;
    hi = w0 + wn;
    return;
  }
  int count = 0;
  source_span(precompute_coeffs(full, out), full, w0, wn, lo, count);
  hi = lo + count;
}

} // namespace

bool resize_source_window(int full_width, int full_height, int out_width,
                          int out_height, int window_x, int window_y,
                          int window_width, int window_height, int &x0, int &y0,
                          int &x1, int &y1) {
  if (full_width <= 0 || full_height <= 0 ||
      !valid_window(out_width, out_height, window_x, window_y, window_width,
                    window_height)) {
    return false;
  }
  axis_span(full_width, out_width, window_x, window_width, x0, x1);
  axis_span(full_height, out_height, window_y, window_height, y0, y1);
  return true;
}

Image8 resize_bilinear(const Image8 &src, int out_width, int out_height) {
  return resize_bilinear_window(src, out_width, out_height, 0, 0, out_width,
                                out_height);
}

Image8 resize_bilinear_window(const Image8 &src, int out_width, int out_height,
                              int window_x, int window_y, int window_width,
                              int window_height) {
  return resize_bilinear_window(src, 0, 0, src.width, src.height, out_width,
                                out_height, window_x, window_y, window_width,
                                window_height);
}

Image8 resize_bilinear_window(const Image8 &src, int region_x, int region_y,
                              int full_width, int full_height, int out_width,
                              int out_height, int window_x, int window_y,
                              int window_width, int window_height) {
  int need_x0 = 0;
  int need_y0 = 0;
  int need_x1 = 0;
  int need_y1 = 0;
  if (!src.valid() || region_x < 0 || region_y < 0 ||
      !resize_source_window(full_width, full_height, out_width, out_height,
                            window_x, window_y, window_width, window_height,
                            need_x0, need_y0, need_x1, need_y1) ||
      need_x0 < region_x || need_y0 < region_y ||
      need_x1 > region_x + src.width || need_y1 > region_y + src.height) {
    return Image8{};
  }

  const bool scale_rows = out_height != full_height;
  AxisCoeffs vertical;
  // Rows the vertical pass will read, so the horizontal pass produces no more
  // than that. For a centre crop this is well under the full height.
  int first_row = 0;
  int row_count = 0;
  if (scale_rows) {
    vertical = precompute_coeffs(full_height, out_height);
    source_span(vertical, full_height, window_y, window_height,
                         first_row, row_count);
  } else {
    first_row = window_y;
    row_count = window_height;
  }

  // Pillow resamples horizontally first and clamps the intermediate image back
  // to 8 bits before the vertical pass. A single fused pass would round
  // differently and would not reproduce the exporter's tensors.
  Image8 horizontal;
  if (out_width == full_width) {
    horizontal = make_image(window_width, row_count, src.channels);
    const std::size_t row_bytes =
        static_cast<std::size_t>(window_width) * src.channels;
    for (int y = 0; y < row_count; ++y) {
      const std::size_t src_offset =
          (static_cast<std::size_t>(first_row + y - region_y) * src.width +
           (window_x - region_x)) *
          src.channels;
      std::copy_n(src.pixels.begin() + static_cast<std::ptrdiff_t>(src_offset),
                  row_bytes,
                  horizontal.pixels.begin() +
                      static_cast<std::ptrdiff_t>(y) *
                          static_cast<std::ptrdiff_t>(row_bytes));
    }
  } else {
    const AxisCoeffs across = precompute_coeffs(full_width, out_width);
    horizontal = resample_horizontal(src, region_x, region_y, across, window_x,
                                     window_width, first_row, row_count);
  }

  if (!scale_rows) {
    return horizontal;
  }
  return resample_vertical(horizontal, vertical, window_y, window_height,
                           first_row);
}

} // namespace dataset
} // namespace deeptile
