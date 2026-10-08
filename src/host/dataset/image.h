/**
 * @file image.h
 * @brief Interleaved 8-bit image container shared by the dataset host path.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_IMAGE_H_
#define DEEPTILE_DATASET_IMAGE_H_

#include <cstdint>
#include <vector>

namespace deeptile {
namespace dataset {

/**
 * Row-major interleaved 8-bit image.
 *
 * Pixels are stored HWC with no row padding, so element (y, x, c) lives at
 * index ((y * width) + x) * channels + c. Channel order is whatever the
 * producer documents; the preprocessing path requires RGB.
 */
struct Image8 {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<std::uint8_t> pixels;

  bool empty() const { return pixels.empty(); }

  /** Expected pixel-buffer length for the current dimensions. */
  std::size_t expected_size() const {
    return static_cast<std::size_t>(height) * static_cast<std::size_t>(width) *
           static_cast<std::size_t>(channels);
  }

  bool valid() const {
    return width > 0 && height > 0 && channels > 0 &&
           pixels.size() == expected_size();
  }
};

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_IMAGE_H_
