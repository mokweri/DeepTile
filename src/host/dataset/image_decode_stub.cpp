/**
 * @file image_decode_stub.cpp
 * @brief Decoder stub for builds without an image-codec backend.
 *
 * Native builds must not require OpenCV, so the deterministic preprocessing
 * path stays testable without it. Callers check image_decoding_available()
 * before indexing a dataset.
 */

#include "dataset/image_decode.h"

namespace deeptile {
namespace dataset {

bool image_decoding_available() { return false; }

bool decode_image_rgb(const std::string &path, Image8 &out,
                      std::string &error) {
  (void)path;
  (void)out;
  error = "this build has no image decoder; rebuild with OpenCV available";
  return false;
}

bool decode_image_rgb_region(const std::string &path, const RegionChooser &,
                             ImageRegion &out, std::string &error) {
  (void)out;
  return decode_image_rgb(path, out.image, error);
}

} // namespace dataset
} // namespace deeptile
