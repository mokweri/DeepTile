/**
 * @file preprocess.cpp
 * @brief Implements exporter-compatible image preprocessing.
 *
 * Arithmetic is deliberately performed in single precision, matching the
 * exporter's tensor pipeline. Widening any step to double changes rounding at
 * the quantization boundary and breaks bit-exact agreement.
 */

#include "dataset/preprocess.h"

#include "dataset/resize.h"
#include "runtime/stem_fold.h"

#include <algorithm>
#include <cmath>

namespace deeptile {
namespace dataset {

bool parse_resize_convention(const std::string &name, ResizeConvention &out) {
  if (name == "export") {
    out = ResizeConvention::ExportSquash;
    return true;
  }
  if (name == "eval") {
    out = ResizeConvention::EvalShortSideCrop;
    return true;
  }
  return false;
}

const char *resize_convention_name(ResizeConvention convention) {
  return convention == ResizeConvention::ExportSquash ? "export" : "eval";
}

namespace {

/** The full resize and the window of it that the convention keeps. */
struct ResizeGeometry {
  int out_w = 0;
  int out_h = 0;
  int left = 0;
  int top = 0;
  int side = 0;
};

bool resize_geometry(int width, int height, const PreprocessConfig &config,
                     ResizeGeometry &g) {
  if (width <= 0 || height <= 0 || config.target_side <= 0) {
    return false;
  }
  g.side = config.target_side;
  if (config.convention == ResizeConvention::ExportSquash) {
    g.out_w = g.out_h = config.target_side;
    return true;
  }
  if (config.short_side < config.target_side) {
    return false;
  }
  // Scale the short side to `short_side` and derive the long side exactly as
  // torchvision does: integer truncation of `size * long / short`, not
  // rounding. The two differ for most aspect ratios and shift the centre crop.
  const double side = static_cast<double>(config.short_side);
  g.out_w = config.short_side;
  g.out_h = config.short_side;
  if (width < height) {
    g.out_h = static_cast<int>(side * height / width);
  } else {
    g.out_w = static_cast<int>(side * width / height);
  }
  // Resize and crop in one step: the centre crop discards most of the resized
  // image, and computing only the surviving window is bit-identical.
  g.left = (g.out_w - config.target_side) / 2;
  g.top = (g.out_h - config.target_side) / 2;
  return true;
}

} // namespace

Image8 apply_resize_convention(const Image8 &rgb,
                               const PreprocessConfig &config) {
  ResizeGeometry g;
  if (!rgb.valid() || !resize_geometry(rgb.width, rgb.height, config, g)) {
    return Image8{};
  }
  return resize_bilinear_window(rgb, g.out_w, g.out_h, g.left, g.top, g.side,
                                g.side);
}

bool preprocess_source_window(int full_width, int full_height,
                              const PreprocessConfig &config, int &x0, int &y0,
                              int &x1, int &y1) {
  ResizeGeometry g;
  return resize_geometry(full_width, full_height, config, g) &&
         resize_source_window(full_width, full_height, g.out_w, g.out_h,
                              g.left, g.top, g.side, g.side, x0, y0, x1, y1);
}

QuantizationTable build_quantization_table(const PreprocessConfig &config) {
  QuantizationTable table;
  const float scale = static_cast<float>(1 << config.frac);
  for (std::size_t c = 0; c < 3; ++c) {
    const float mean = config.mean[c];
    const float stddev = config.stddev[c];
    for (int v = 0; v < 256; ++v) {
      // Identical arithmetic to the per-pixel path, so the table is exact.
      const float normalized =
          ((static_cast<float>(v) / 255.0F) - mean) / stddev;
      const float q = std::floor(normalized * scale + 0.5F);
      table.lanes[c][static_cast<std::size_t>(v)] =
          static_cast<std::int8_t>(std::min(127.0F, std::max(-128.0F, q)));
    }
  }
  return table;
}

std::vector<std::uint8_t> preprocess_rgb_interleaved(const Image8 &rgb,
                                                     const PreprocessConfig &config,
                                                     int lanes) {
  if (config.frac < 0) {
    return {};
  }
  return preprocess_rgb_interleaved(rgb, config, build_quantization_table(config),
                                    lanes);
}

std::vector<std::uint8_t> preprocess_rgb_interleaved(const Image8 &rgb,
                                                     const PreprocessConfig &config,
                                                     const QuantizationTable &table,
                                                     int lanes) {
  if (!rgb.valid() || rgb.channels != 3 || config.frac < 0 || lanes < 3) {
    return {};
  }
  return preprocess_rgb_interleaved(rgb, 0, 0, rgb.width, rgb.height, config,
                                    table, lanes);
}

std::vector<std::uint8_t> preprocess_rgb_interleaved(
    const Image8 &region, int region_x, int region_y, int full_width,
    int full_height, const PreprocessConfig &config,
    const QuantizationTable &table, int lanes) {
  ResizeGeometry g;
  if (!region.valid() || region.channels != 3 || config.frac < 0 || lanes < 3 ||
      !resize_geometry(full_width, full_height, config, g)) {
    return {};
  }
  const Image8 sized =
      resize_bilinear_window(region, region_x, region_y, full_width,
                             full_height, g.out_w, g.out_h, g.left, g.top,
                             g.side, g.side);
  if (!sized.valid()) {
    return {};
  }

  const std::size_t stride = static_cast<std::size_t>(lanes);
  const std::int8_t *lut_r = table.lanes[0].data();
  const std::int8_t *lut_g = table.lanes[1].data();
  const std::int8_t *lut_b = table.lanes[2].data();

  if (config.fold_pad >= 0) {
    // Space-to-depth for a folded stem: each 2x2 block of the zero-bordered
    // image becomes one pixel of 12 lanes, in stem_fold_lane() order.
    if (lanes < 12) {
      return {};
    }
    const int side = config.target_side;
    const int pad = config.fold_pad;
    const int folded = (side + 2 * pad + (side + 2 * pad) % 2) / 2;
    std::vector<std::uint8_t> out(
        static_cast<std::size_t>(folded) * folded * stride, 0);
    const std::int8_t *luts[3] = {lut_r, lut_g, lut_b};
    for (int i = 0; i < folded; ++i) {
      for (int dy = 0; dy < 2; ++dy) {
        const int y = 2 * i + dy - pad;
        if (y < 0 || y >= side) {
          continue;
        }
        for (int j = 0; j < folded; ++j) {
          std::uint8_t *dst =
              out.data() + (static_cast<std::size_t>(i) * folded + j) * stride;
          for (int dx = 0; dx < 2; ++dx) {
            const int x = 2 * j + dx - pad;
            if (x < 0 || x >= side) {
              continue;
            }
            const std::uint8_t *src =
                sized.pixels.data() + (static_cast<std::size_t>(y) * side + x) * 3;
            for (int c = 0; c < 3; ++c) {
              dst[deeptile::runtime::stem_fold_lane(dy, dx, c, 3)] =
                  static_cast<std::uint8_t>(luts[c][src[c]]);
            }
          }
        }
      }
    }
    return out;
  }

  const std::size_t pixels =
      static_cast<std::size_t>(config.target_side) * config.target_side;
  // Unused lanes stay zero, matching the packer's padding.
  std::vector<std::uint8_t> out(pixels * stride, 0);

  const std::uint8_t *src = sized.pixels.data();
  std::uint8_t *dst = out.data();
  for (std::size_t i = 0; i < pixels; ++i, src += 3, dst += stride) {
    dst[0] = static_cast<std::uint8_t>(lut_r[src[0]]);
    dst[1] = static_cast<std::uint8_t>(lut_g[src[1]]);
    dst[2] = static_cast<std::uint8_t>(lut_b[src[2]]);
  }
  return out;
}

std::vector<std::int8_t> quantize_chw(const std::vector<float> &chw, int frac) {
  const float scale = static_cast<float>(1 << frac);
  std::vector<std::int8_t> out(chw.size());
  for (std::size_t i = 0; i < chw.size(); ++i) {
    const float q = std::floor(chw[i] * scale + 0.5F);
    out[i] = static_cast<std::int8_t>(std::min(127.0F, std::max(-128.0F, q)));
  }
  return out;
}

std::vector<std::int8_t> preprocess_rgb(const Image8 &rgb,
                                        const PreprocessConfig &config) {
  if (!rgb.valid() || rgb.channels != 3 || config.frac < 0) {
    return {};
  }
  const Image8 sized = apply_resize_convention(rgb, config);
  if (!sized.valid()) {
    return {};
  }

  const int side = config.target_side;
  const std::size_t plane = static_cast<std::size_t>(side) * side;
  const QuantizationTable table = build_quantization_table(config);

  std::vector<std::int8_t> out(plane * 3);
  for (std::size_t c = 0; c < 3; ++c) {
    const auto &lane = table.lanes[c];
    std::int8_t *plane_out = out.data() + c * plane;
    for (std::size_t i = 0; i < plane; ++i) {
      plane_out[i] = lane[sized.pixels[i * 3 + c]];
    }
  }
  return out;
}

} // namespace dataset
} // namespace deeptile
