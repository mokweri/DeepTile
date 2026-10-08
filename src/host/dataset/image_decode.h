/**
 * @file image_decode.h
 * @brief Image file decoding for the dataset path.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_IMAGE_DECODE_H_
#define DEEPTILE_DATASET_IMAGE_DECODE_H_

#include "dataset/image.h"

#include <functional>
#include <string>

namespace deeptile {
namespace dataset {

/** True when this build was compiled with a decoder backend available. */
bool image_decoding_available();

/**
 * Decode an image file to 3-channel RGB.
 *
 * Grayscale and CMYK sources are converted, so the result is always 3-channel
 * regardless of how the file was stored; the validation set contains both.
 *
 * @param path Image file path.
 * @param out Receives the decoded image on success.
 * @param error Receives a human-readable reason on failure.
 * @return False if the file is missing, unreadable, or unsupported.
 */
bool decode_image_rgb(const std::string &path, Image8 &out, std::string &error);

/** Decoded pixels covering a rectangle of a larger image. */
struct ImageRegion {
  Image8 image;         ///< Pixels from (x, y) of the full image.
  int x = 0;
  int y = 0;
  int full_width = 0;   ///< Dimensions of the whole image.
  int full_height = 0;
};

/**
 * Chooses the rectangle [x0, x1) x [y0, y1) to decode, given the image size.
 */
using RegionChooser = std::function<bool(int full_width, int full_height,
                                         int &x0, int &y0, int &x1, int &y1)>;

/**
 * Decode at least the rectangle `choose` asks for, as 3-channel RGB.
 *
 * Every pixel inside the rectangle equals decode_image_rgb() of the same file.
 * Baseline and progressive JPEGs are decoded by libjpeg directly, skipping rows
 * and columns outside the rectangle (keeping one block row and column of
 * context, which the upsampler reads). Files that path cannot reproduce
 * exactly, such as an EXIF rotation, CMYK or a non-JPEG, are decoded in full
 * by decode_image_rgb(), and the region is then the whole image.
 */
bool decode_image_rgb_region(const std::string &path,
                             const RegionChooser &choose, ImageRegion &out,
                             std::string &error);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_IMAGE_DECODE_H_
