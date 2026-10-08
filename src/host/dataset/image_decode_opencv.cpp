/**
 * @file image_decode_opencv.cpp
 * @brief OpenCV-backed image decoding, with a libjpeg region decoder.
 *
 * OpenCV is used only as a decoder. Its resizing does not reproduce the
 * exporter's Pillow output, so scaling stays in dataset/resize.cpp and this
 * translation unit must not grow an imgproc dependency.
 *
 * decode_image_rgb() is the reference: OpenCV, then BGR to RGB. The region
 * decoder calls libjpeg-turbo (OpenCV's own JPEG backend) with the settings
 * OpenCV uses, so it produces the same pixels, and decodes only the rows and
 * columns preprocessing reads. Whatever it cannot reproduce exactly goes back
 * to the reference.
 */

#include "dataset/image_decode.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef DEEPTILE_HAVE_LIBJPEG
#include <csetjmp>
#include <jpeglib.h>
#endif

namespace deeptile {
namespace dataset {

bool image_decoding_available() { return true; }

bool decode_image_rgb(const std::string &path, Image8 &out,
                      std::string &error) {
  // IMREAD_COLOR normalizes grayscale and CMYK sources to 3-channel BGR.
  const cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    error = "failed to decode image: " + path;
    return false;
  }
  if (bgr.type() != CV_8UC3) {
    error = "unexpected decoded pixel format: " + path;
    return false;
  }

  out.width = bgr.cols;
  out.height = bgr.rows;
  out.channels = 3;
  out.pixels.resize(out.expected_size());

  // Convert BGR to RGB during the copy; the exporter normalizes RGB channels.
  for (int y = 0; y < bgr.rows; ++y) {
    const std::uint8_t *src = bgr.ptr<std::uint8_t>(y);
    std::uint8_t *dst =
        out.pixels.data() + static_cast<std::size_t>(y) * bgr.cols * 3;
    for (int x = 0; x < bgr.cols; ++x) {
      dst[x * 3 + 0] = src[x * 3 + 2];
      dst[x * 3 + 1] = src[x * 3 + 1];
      dst[x * 3 + 2] = src[x * 3 + 0];
    }
  }
  return true;
}

namespace {

bool decode_full_region(const std::string &path, ImageRegion &out,
                        std::string &error) {
  if (!decode_image_rgb(path, out.image, error)) {
    return false;
  }
  out.x = 0;
  out.y = 0;
  out.full_width = out.image.width;
  out.full_height = out.image.height;
  return true;
}

#ifdef DEEPTILE_HAVE_LIBJPEG

struct ErrorJump {
  jpeg_error_mgr manager;
  std::jmp_buf jump;
};

void on_error(j_common_ptr cinfo) {
  std::longjmp(reinterpret_cast<ErrorJump *>(cinfo->err)->jump, 1);
}

// Corrupt-data warnings do not stop decoding, as in OpenCV; they are not printed.
void on_message(j_common_ptr) {}

std::uint32_t read_u32(const std::uint8_t *p, bool little) {
  return little ? (std::uint32_t{p[0]} | std::uint32_t{p[1]} << 8 |
                   std::uint32_t{p[2]} << 16 | std::uint32_t{p[3]} << 24)
                : (std::uint32_t{p[3]} | std::uint32_t{p[2]} << 8 |
                   std::uint32_t{p[1]} << 16 | std::uint32_t{p[0]} << 24);
}

std::uint16_t read_u16(const std::uint8_t *p, bool little) {
  return little ? static_cast<std::uint16_t>(p[0] | p[1] << 8)
                : static_cast<std::uint16_t>(p[1] | p[0] << 8);
}

/**
 * EXIF orientation from the APP1 markers: 1 when absent, 0 when present but
 * unreadable. OpenCV's imread rotates the image by this tag, and only an
 * absent or upright tag leaves pixels where libjpeg puts them.
 */
int exif_orientation(const jpeg_saved_marker_ptr markers) {
  for (jpeg_saved_marker_ptr m = markers; m != nullptr; m = m->next) {
    if (m->marker != JPEG_APP0 + 1 || m->data_length < 14 ||
        std::memcmp(m->data, "Exif\0\0", 6) != 0) {
      continue;
    }
    const std::uint8_t *tiff = m->data + 6;
    const std::size_t size = m->data_length - 6;
    bool little = false;
    if (tiff[0] == 'I' && tiff[1] == 'I') {
      little = true;
    } else if (!(tiff[0] == 'M' && tiff[1] == 'M')) {
      return 0;
    }
    const std::uint32_t ifd = read_u32(tiff + 4, little);
    if (ifd + 2 > size) {
      return 0;
    }
    const std::uint16_t entries = read_u16(tiff + ifd, little);
    for (std::uint32_t i = 0; i < entries; ++i) {
      const std::size_t e = ifd + 2 + 12 * static_cast<std::size_t>(i);
      if (e + 12 > size) {
        return 0;
      }
      if (read_u16(tiff + e, little) == 0x0112) {
        return read_u16(tiff + e + 8, little);
      }
    }
  }
  return 1;
}

/** Outcome of the libjpeg path: decoded, or hand the file to the reference. */
enum class JpegResult { Decoded, Fallback };

JpegResult decode_jpeg_region(const std::string &path,
                              const RegionChooser &choose, ImageRegion &out) {
  std::FILE *file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return JpegResult::Fallback;
  }
  jpeg_decompress_struct cinfo;
  ErrorJump err;
  cinfo.err = jpeg_std_error(&err.manager);
  err.manager.error_exit = on_error;
  err.manager.output_message = on_message;
  // Everything the decode writes after setjmp lives in `out` or the heap, so
  // nothing needs to survive the jump but the file and the decompressor.
  if (setjmp(err.jump)) {
    jpeg_destroy_decompress(&cinfo);
    std::fclose(file);
    return JpegResult::Fallback;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, file);
  jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
  jpeg_read_header(&cinfo, TRUE);

  const bool plain = cinfo.data_precision == 8 &&
                     cinfo.jpeg_color_space != JCS_CMYK &&
                     cinfo.jpeg_color_space != JCS_YCCK &&
                     exif_orientation(cinfo.marker_list) == 1;
  int x0 = 0;
  int y0 = 0;
  int x1 = 0;
  int y1 = 0;
  if (!plain || !choose(static_cast<int>(cinfo.image_width),
                        static_cast<int>(cinfo.image_height), x0, y0, x1, y1)) {
    jpeg_destroy_decompress(&cinfo);
    std::fclose(file);
    return JpegResult::Fallback;
  }

  // OpenCV's settings for a 3-channel result: default IDCT and upsampling,
  // libjpeg-turbo's own colour conversion.
  cinfo.out_color_space = JCS_EXT_RGB;
  jpeg_start_decompress(&cinfo);
  const int width = static_cast<int>(cinfo.output_width);
  const int height = static_cast<int>(cinfo.output_height);

  // One block of context on every side: the upsampler reads neighbouring
  // chroma samples, and with them the region equals a full decode.
  const int pad_x = cinfo.max_h_samp_factor * DCTSIZE;
  const int pad_y = cinfo.max_v_samp_factor * DCTSIZE;
  JDIMENSION crop_x = static_cast<JDIMENSION>(std::max(0, x0 - pad_x));
  JDIMENSION crop_w =
      static_cast<JDIMENSION>(std::min(width, x1 + pad_x)) - crop_x;
  jpeg_crop_scanline(&cinfo, &crop_x, &crop_w);
  const int row0 = std::max(0, y0 - pad_y);
  const int row1 = std::min(height, y1 + pad_y);
  if (row0 > 0) {
    jpeg_skip_scanlines(&cinfo, static_cast<JDIMENSION>(row0));
  }

  out.image.width = static_cast<int>(cinfo.output_width);
  out.image.height = row1 - row0;
  out.image.channels = 3;
  out.image.pixels.resize(out.image.expected_size());
  const std::size_t row_bytes = static_cast<std::size_t>(out.image.width) * 3;
  for (int y = 0; y < out.image.height; ++y) {
    JSAMPROW row = out.image.pixels.data() + static_cast<std::size_t>(y) * row_bytes;
    jpeg_read_scanlines(&cinfo, &row, 1);
  }
  out.x = static_cast<int>(crop_x);
  out.y = row0;
  out.full_width = width;
  out.full_height = height;

  // The remaining rows are not needed; abort rather than decode them.
  jpeg_abort_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  std::fclose(file);
  return JpegResult::Decoded;
}

#endif // DEEPTILE_HAVE_LIBJPEG

} // namespace

bool decode_image_rgb_region(const std::string &path,
                             const RegionChooser &choose, ImageRegion &out,
                             std::string &error) {
#ifdef DEEPTILE_HAVE_LIBJPEG
  if (decode_jpeg_region(path, choose, out) == JpegResult::Decoded) {
    return true;
  }
#else
  (void)choose;
#endif
  return decode_full_region(path, out, error);
}

} // namespace dataset
} // namespace deeptile
