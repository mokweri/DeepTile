/**
 * @file preprocess.h
 * @brief Image-to-int8 preprocessing matching the exporter's convention.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_PREPROCESS_H_
#define DEEPTILE_DATASET_PREPROCESS_H_

#include "dataset/image.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace deeptile {
namespace dataset {

/**
 * Resize geometry used to turn a source image into a square network input.
 *
 * The exporter and the accuracy pipeline currently disagree, so the convention
 * is always selected explicitly and recorded with the run. Defaulting it would
 * silently change every tensor.
 */
enum class ResizeConvention {
  /** Resize directly to the target side, distorting aspect ratio. */
  ExportSquash,
  /** Resize the short side, then centre-crop to the target side. */
  EvalShortSideCrop,
};

/** Parse a convention name ("export" or "eval"); returns false if unknown. */
bool parse_resize_convention(const std::string &name, ResizeConvention &out);

/** Canonical name of a convention, suitable for a run record. */
const char *resize_convention_name(ResizeConvention convention);

/**
 * Preprocessing parameters.
 *
 * Defaults reproduce the checked-in ImageNet model packages: torchvision
 * normalization and a Q2.5 input tensor. `frac` must match the graph's input
 * tensor, and `short_side` applies only to EvalShortSideCrop.
 */
struct PreprocessConfig {
  ResizeConvention convention = ResizeConvention::EvalShortSideCrop;
  int target_side = 224;
  int short_side = 256;
  std::array<float, 3> mean = {0.485F, 0.456F, 0.406F};
  std::array<float, 3> stddev = {0.229F, 0.224F, 0.225F};
  int frac = 5;
  /// When non-negative, write the input folded by 2x2 space-to-depth with
  /// this zero border, matching a stem rewritten by runtime/stem_fold.h.
  int fold_pad = -1;
};

/**
 * Per-channel mapping from source byte to quantized output.
 *
 * Normalization and quantization depend only on the channel and the 8-bit
 * sample, so the whole chain collapses to 768 precomputed entries. This is not
 * an approximation: the table is built with the same arithmetic it replaces.
 */
struct QuantizationTable {
  std::array<std::array<std::int8_t, 256>, 3> lanes{};
};

/** Build the table implied by a configuration's normalization and frac. */
QuantizationTable build_quantization_table(const PreprocessConfig &config);

/**
 * Preprocess into pixel-interleaved output padded to `lanes` channels.
 *
 * Layout is [H][W][lanes] with unused lanes zeroed, which is what the
 * accelerator consumes when a tensor fits in one channel group. Producing it
 * directly avoids materializing an intermediate CHW tensor and repacking it.
 *
 * @param rgb Decoded image; must be valid interleaved RGB.
 * @param config Resize, normalization, and quantization parameters.
 * @param lanes Channels per pixel in the output; must be at least the image's.
 * @return Packed bytes, or empty if the input or lane count is unusable.
 */
std::vector<std::uint8_t> preprocess_rgb_interleaved(const Image8 &rgb,
                                                     const PreprocessConfig &config,
                                                     int lanes);

/**
 * As above, with a table built once by build_quantization_table(config).
 * A caller preprocessing many images under one configuration should use this
 * form; the output is identical.
 */
std::vector<std::uint8_t> preprocess_rgb_interleaved(const Image8 &rgb,
                                                     const PreprocessConfig &config,
                                                     const QuantizationTable &table,
                                                     int lanes);

/**
 * The rectangle [x0, x1) x [y0, y1) of a full_width x full_height image that
 * preprocessing reads; a decoder may produce only this region.
 *
 * @return False if the dimensions or configuration are unusable.
 */
bool preprocess_source_window(int full_width, int full_height,
                              const PreprocessConfig &config, int &x0, int &y0,
                              int &x1, int &y1);

/**
 * preprocess_rgb_interleaved() on a decoded region of the image.
 *
 * `region` holds pixels from (region_x, region_y) of a full_width x
 * full_height image and must cover preprocess_source_window(); the output is
 * identical to preprocessing the whole image.
 */
std::vector<std::uint8_t> preprocess_rgb_interleaved(
    const Image8 &region, int region_x, int region_y, int full_width,
    int full_height, const PreprocessConfig &config,
    const QuantizationTable &table, int lanes);

/**
 * Convert a decoded RGB image into the graph's int8 CHW input tensor.
 *
 * Applies the configured resize geometry, scales to [0,1], normalizes, and
 * quantizes with round-half-up as the exporter does. Output is CHW with
 * `target_side` rows and columns and one plane per input channel.
 *
 * @param rgb Decoded image; must be 3-channel RGB and satisfy Image8::valid().
 * @param config Resize, normalization, and quantization parameters.
 * @return Quantized tensor, or an empty vector if the input is unusable.
 */
std::vector<std::int8_t> preprocess_rgb(const Image8 &rgb,
                                        const PreprocessConfig &config);

/** Apply only the configured resize and crop, returning 8-bit RGB. */
Image8 apply_resize_convention(const Image8 &rgb,
                               const PreprocessConfig &config);

/**
 * Quantize normalized float CHW data to int8.
 *
 * Uses round-half-up, `floor(x * 2^frac + 0.5)`, matching the exporter. This
 * is not std::round, which rounds halves away from zero and therefore differs
 * on every negative half-step.
 */
std::vector<std::int8_t> quantize_chw(const std::vector<float> &chw, int frac);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_PREPROCESS_H_
