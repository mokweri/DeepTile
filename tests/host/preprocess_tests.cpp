/**
 * @file preprocess_tests.cpp
 * @brief Bit-exactness tests for host image preprocessing.
 *
 * Expected outputs are produced by Pillow through
 * scripts/tools/gen_preprocess_fixtures.py. Inputs are synthesized here by an
 * LCG that mirrors the generator, so only expectations are checked in.
 *
 * These tests need no image decoder, so the deterministic part of the
 * preprocessing contract is covered in every native build. Decoding from JPEG
 * is covered by the anchor test under tests/integration/.
 */

#include "dataset/preprocess.h"
#include "dataset/resize.h"
#include "runtime/runtime.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "    ASSERT FAILED: " << msg << "\n";                       \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace {

using deeptile::dataset::Image8;
using deeptile::dataset::PreprocessConfig;
using deeptile::dataset::ResizeConvention;

std::filesystem::path fixture_dir() {
#ifdef DEEPTILE_REPO_ROOT
  return std::filesystem::path(DEEPTILE_REPO_ROOT) / "tests" / "host" / "data" /
         "preprocess";
#else
  return std::filesystem::current_path();
#endif
}

/** Mirror of synth() in gen_preprocess_fixtures.py. */
Image8 synth_image(int width, int height, std::uint32_t seed) {
  Image8 img;
  img.width = width;
  img.height = height;
  img.channels = 3;
  img.pixels.resize(img.expected_size());

  std::uint32_t state = seed;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        state = state * 1664525U + 1013904223U;
        const int noise = static_cast<int>((state >> 16) & 0x3FU) - 32;
        int base = 0;
        if (c == 0) {
          base = 255 * x / (width - 1 > 0 ? width - 1 : 1);
        } else if (c == 1) {
          base = 255 * y / (height - 1 > 0 ? height - 1 : 1);
        } else {
          base = 255 * ((x / 3 + y / 5) % 2);
        }
        int v = base + noise;
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
        const std::size_t i =
            (static_cast<std::size_t>(y) * width + x) * 3 + c;
        img.pixels[i] = static_cast<std::uint8_t>(v);
      }
    }
  }
  return img;
}

bool read_ppm(const std::filesystem::path &path, Image8 &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::string magic;
  int width = 0;
  int height = 0;
  int maxval = 0;
  in >> magic >> width >> height >> maxval;
  if (magic != "P6" || maxval != 255) {
    return false;
  }
  in.get();  // single whitespace byte before the pixel data
  out.width = width;
  out.height = height;
  out.channels = 3;
  out.pixels.resize(out.expected_size());
  in.read(reinterpret_cast<char *>(out.pixels.data()),
          static_cast<std::streamsize>(out.pixels.size()));
  return in.good() || in.eof();
}

bool read_int8(const std::filesystem::path &path,
               std::vector<std::int8_t> &out) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    return false;
  }
  out.resize(static_cast<std::size_t>(in.tellg()));
  in.seekg(0);
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(out.size()));
  return true;
}

/** Report the first differing element so a regression is locatable. */
bool compare_bytes(const std::vector<std::uint8_t> &got,
                   const std::vector<std::uint8_t> &want,
                   const std::string &label) {
  TEST_ASSERT(got.size() == want.size(),
              label << " size " << got.size() << " != " << want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      std::cerr << "    " << label << " first diff at " << i << ": got "
                << static_cast<int>(got[i]) << " want "
                << static_cast<int>(want[i]) << "\n";
      return false;
    }
  }
  return true;
}

bool test_resize_case(const std::string &name, int w, int h, std::uint32_t seed,
                      int ow, int oh) {
  Image8 expected;
  TEST_ASSERT(read_ppm(fixture_dir() / (name + "_out.ppm"), expected),
              "cannot read fixture for " << name);
  const Image8 got =
      deeptile::dataset::resize_bilinear(synth_image(w, h, seed), ow, oh);
  TEST_ASSERT(got.valid(), name << " produced an invalid image");
  TEST_ASSERT(got.width == expected.width && got.height == expected.height,
              name << " dimensions differ");
  return compare_bytes(got.pixels, expected.pixels, name);
}

bool test_convention(ResizeConvention convention, const std::string &fixture) {
  std::vector<std::int8_t> expected;
  TEST_ASSERT(read_int8(fixture_dir() / fixture, expected),
              "cannot read fixture " << fixture);

  PreprocessConfig config;
  config.convention = convention;
  const std::vector<std::int8_t> got =
      deeptile::dataset::preprocess_rgb(synth_image(400, 300, 99), config);

  TEST_ASSERT(got.size() == expected.size(),
              fixture << " size " << got.size() << " != " << expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      std::cerr << "    " << fixture << " first diff at " << i << ": got "
                << static_cast<int>(got[i]) << " want "
                << static_cast<int>(expected[i]) << "\n";
      return false;
    }
  }
  return true;
}

/** Round-half-up differs from std::round on negative halves; pin it. */
bool test_quantize_rounding() {
  const std::vector<float> values = {0.5F / 32, -0.5F / 32, 1.5F / 32,
                                     -1.5F / 32, 0.0F, 100.0F, -100.0F};
  const std::vector<int> expected = {1, 0, 2, -1, 0, 127, -128};
  const std::vector<std::int8_t> got = deeptile::dataset::quantize_chw(values, 5);
  TEST_ASSERT(got.size() == expected.size(), "quantize size");
  for (std::size_t i = 0; i < got.size(); ++i) {
    TEST_ASSERT(static_cast<int>(got[i]) == expected[i],
                "quantize[" << i << "] got " << static_cast<int>(got[i])
                            << " want " << expected[i]);
  }
  return true;
}

bool test_convention_names() {
  ResizeConvention parsed = ResizeConvention::EvalShortSideCrop;
  TEST_ASSERT(deeptile::dataset::parse_resize_convention("export", parsed) &&
                  parsed == ResizeConvention::ExportSquash,
              "parse export");
  TEST_ASSERT(deeptile::dataset::parse_resize_convention("eval", parsed) &&
                  parsed == ResizeConvention::EvalShortSideCrop,
              "parse eval");
  TEST_ASSERT(!deeptile::dataset::parse_resize_convention("squash", parsed),
              "unknown convention must be rejected");
  TEST_ASSERT(std::string(deeptile::dataset::resize_convention_name(
                  ResizeConvention::ExportSquash)) == "export",
              "export name");
  return true;
}

/** The two conventions must not silently produce the same tensor. */
bool test_conventions_differ() {
  const Image8 src = synth_image(400, 300, 99);
  PreprocessConfig a;
  a.convention = ResizeConvention::ExportSquash;
  PreprocessConfig b;
  b.convention = ResizeConvention::EvalShortSideCrop;
  const auto ta = deeptile::dataset::preprocess_rgb(src, a);
  const auto tb = deeptile::dataset::preprocess_rgb(src, b);
  TEST_ASSERT(!ta.empty() && ta.size() == tb.size(), "convention sizes");
  TEST_ASSERT(ta != tb, "conventions produced identical tensors");
  return true;
}

bool test_rejects_bad_input() {
  PreprocessConfig config;
  TEST_ASSERT(deeptile::dataset::preprocess_rgb(Image8{}, config).empty(),
              "empty image must be rejected");
  Image8 grayscale = synth_image(64, 64, 7);
  grayscale.channels = 1;  // pixel buffer no longer matches the shape
  TEST_ASSERT(deeptile::dataset::preprocess_rgb(grayscale, config).empty(),
              "inconsistent image must be rejected");
  return true;
}

/**
 * A window of a resize must equal the same window of the full resize.
 *
 * This is what allows the centre crop to be fused into resizing, so it is
 * checked directly rather than only through the convention fixtures.
 */
bool test_resize_window_matches_full() {
  const Image8 src = synth_image(123, 91, 77);
  const int ow = 71;
  const int oh = 53;
  const Image8 full = deeptile::dataset::resize_bilinear(src, ow, oh);
  TEST_ASSERT(full.valid(), "full resize invalid");

  const int windows[][4] = {{0, 0, 71, 53}, {10, 7, 32, 24},
                            {40, 30, 31, 23}, {35, 26, 1, 1}};
  for (const auto &w : windows) {
    const Image8 got = deeptile::dataset::resize_bilinear_window(
        src, ow, oh, w[0], w[1], w[2], w[3]);
    TEST_ASSERT(got.valid(), "window resize invalid");
    TEST_ASSERT(got.width == w[2] && got.height == w[3], "window dimensions");
    for (int y = 0; y < w[3]; ++y) {
      for (int x = 0; x < w[2]; ++x) {
        for (int c = 0; c < 3; ++c) {
          const std::size_t gi =
              (static_cast<std::size_t>(y) * w[2] + x) * 3 + c;
          const std::size_t fi =
              (static_cast<std::size_t>(w[1] + y) * ow + (w[0] + x)) * 3 + c;
          TEST_ASSERT(got.pixels[gi] == full.pixels[fi],
                      "window differs at (" << x << "," << y << "," << c << ")");
        }
      }
    }
  }
  return true;
}

/** Out-of-range windows must be rejected, not silently clamped. */
bool test_resize_window_rejects_bad_range() {
  const Image8 src = synth_image(40, 30, 5);
  TEST_ASSERT(
      deeptile::dataset::resize_bilinear_window(src, 20, 15, 10, 0, 20, 5).empty(),
      "window past right edge must be rejected");
  TEST_ASSERT(
      deeptile::dataset::resize_bilinear_window(src, 20, 15, 0, -1, 5, 5).empty(),
      "negative origin must be rejected");
  return true;
}

/**
 * The interleaved fast path must equal packing the CHW result.
 *
 * These are two routes to the same bytes; if they diverge, the accelerator
 * silently receives different input from the reference path.
 */
bool test_interleaved_matches_packed_chw() {
  const Image8 src = synth_image(400, 300, 99);
  for (const auto convention :
       {ResizeConvention::EvalShortSideCrop, ResizeConvention::ExportSquash}) {
    PreprocessConfig config;
    config.convention = convention;

    const std::vector<std::int8_t> chw =
        deeptile::dataset::preprocess_rgb(src, config);
    TEST_ASSERT(!chw.empty(), "CHW path produced nothing");
    const std::vector<int> shape = {3, config.target_side, config.target_side};
    const std::vector<std::uint8_t> expected =
        deeptile::runtime::pack_activation_chw(chw, shape);

    const std::vector<std::uint8_t> got =
        deeptile::dataset::preprocess_rgb_interleaved(src, config, 16);
    TEST_ASSERT(got.size() == expected.size(),
                "interleaved size " << got.size() << " != " << expected.size());
    if (!compare_bytes(got, expected, "interleaved")) {
      return false;
    }
  }
  return true;
}

/** The table must reproduce the arithmetic it replaces, for every byte. */
bool test_quantization_table_matches_arithmetic() {
  PreprocessConfig config;
  const auto table = deeptile::dataset::build_quantization_table(config);
  for (std::size_t c = 0; c < 3; ++c) {
    for (int v = 0; v < 256; ++v) {
      const float normalized =
          ((static_cast<float>(v) / 255.0F) - config.mean[c]) / config.stddev[c];
      const std::vector<std::int8_t> reference =
          deeptile::dataset::quantize_chw({normalized}, config.frac);
      TEST_ASSERT(table.lanes[c][static_cast<std::size_t>(v)] == reference[0],
                  "table[" << c << "][" << v << "] = "
                           << static_cast<int>(
                                  table.lanes[c][static_cast<std::size_t>(v)])
                           << " want " << static_cast<int>(reference[0]));
    }
  }
  return true;
}

bool test_interleaved_rejects_bad_lanes() {
  PreprocessConfig config;
  const Image8 src = synth_image(64, 64, 3);
  TEST_ASSERT(
      deeptile::dataset::preprocess_rgb_interleaved(src, config, 2).empty(),
      "lane count below channel count must be rejected");
  return true;
}

int passed = 0;
int failed = 0;

void run(const std::string &name, const std::function<bool()> &fn) {
  std::cout << "  " << name << " ... " << std::flush;
  if (fn()) {
    std::cout << "PASS\n";
    ++passed;
  } else {
    std::cout << "FAIL\n";
    ++failed;
  }
}

} // namespace

int main() {
  run("resize_downscale",
      [] { return test_resize_case("downscale", 97, 61, 11, 40, 24); });
  run("resize_upscale",
      [] { return test_resize_case("upscale", 17, 23, 22, 40, 40); });
  run("resize_width_only",
      [] { return test_resize_case("width_only", 64, 48, 33, 32, 48); });
  run("resize_passthrough",
      [] { return test_resize_case("passthrough", 33, 21, 44, 33, 21); });
  run("resize_odd_ratio",
      [] { return test_resize_case("odd_ratio", 123, 45, 55, 37, 29); });
  run("convention_export", [] {
    return test_convention(ResizeConvention::ExportSquash,
                           "convention_export.int8.bin");
  });
  run("convention_eval", [] {
    return test_convention(ResizeConvention::EvalShortSideCrop,
                           "convention_eval.int8.bin");
  });
  run("conventions_differ", test_conventions_differ);
  run("quantize_rounding", test_quantize_rounding);
  run("convention_names", test_convention_names);
  run("rejects_bad_input", test_rejects_bad_input);
  run("resize_window_matches_full", test_resize_window_matches_full);
  run("resize_window_rejects_bad_range", test_resize_window_rejects_bad_range);
  run("interleaved_matches_packed_chw", test_interleaved_matches_packed_chw);
  run("quantization_table_matches_arithmetic",
      test_quantization_table_matches_arithmetic);
  run("interleaved_rejects_bad_lanes", test_interleaved_rejects_bad_lanes);

  std::cout << "Preprocess tests: " << passed << " passed, " << failed
            << " failed\n";
  return failed == 0 ? 0 : 1;
}
