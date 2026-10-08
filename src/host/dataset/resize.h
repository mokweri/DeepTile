/**
 * @file resize.h
 * @brief Pillow-compatible antialiased bilinear image resizing.
 * @ingroup deeptile_host
 */

#ifndef DEEPTILE_DATASET_RESIZE_H_
#define DEEPTILE_DATASET_RESIZE_H_

#include "dataset/image.h"

namespace deeptile {
namespace dataset {

/**
 * Resize an interleaved 8-bit image, reproducing Pillow bit-exactly.
 *
 * The exporter preprocesses with torchvision, which resizes through Pillow.
 * Reproducing that output exactly is what allows accelerator predictions to be
 * compared against the reference emulator per image rather than only in
 * aggregate, so this must not be replaced by a general-purpose resize.
 * OpenCV's interpolation modes do not satisfy this contract.
 *
 * Antialiasing is applied when downscaling and disabled when upscaling, again
 * matching Pillow. Any channel count is accepted; channels are independent.
 *
 * @param src Source image; must satisfy Image8::valid().
 * @param out_width Target width in pixels, greater than zero.
 * @param out_height Target height in pixels, greater than zero.
 * @return Resized image, or an empty image if an argument is invalid.
 */
Image8 resize_bilinear(const Image8 &src, int out_width, int out_height);

/**
 * Resize and return only a window of the result.
 *
 * Each output pixel depends solely on its own coordinate, so computing a
 * subset is bit-identical to resizing fully and cropping afterwards. The
 * evaluation convention discards most of a resized image, and the saved work
 * is proportional to the discarded area.
 *
 * @param src Source image; must satisfy Image8::valid().
 * @param out_width Width of the full resize this window is taken from.
 * @param out_height Height of the full resize this window is taken from.
 * @param window_x Left edge of the window within that full resize.
 * @param window_y Top edge of the window within that full resize.
 * @param window_width Window width; must fit within the full resize.
 * @param window_height Window height; must fit within the full resize.
 */
Image8 resize_bilinear_window(const Image8 &src, int out_width, int out_height,
                              int window_x, int window_y, int window_width,
                              int window_height);

/**
 * The source rectangle [x0, x1) x [y0, y1) that a window of a resize reads.
 *
 * A decoder that produces only this rectangle saves the work of decoding the
 * rest, and the window computed from it is identical.
 *
 * @param full_width Width of the whole source image.
 * @param full_height Height of the whole source image.
 * @param out_width, out_height, window_x, window_y, window_width, window_height
 *        As for resize_bilinear_window().
 * @param[out] x0, y0 Inclusive top-left corner of the source rectangle.
 * @param[out] x1, y1 Exclusive bottom-right corner of the source rectangle.
 * @return False if an argument is invalid.
 */
bool resize_source_window(int full_width, int full_height, int out_width,
                          int out_height, int window_x, int window_y,
                          int window_width, int window_height, int &x0, int &y0,
                          int &x1, int &y1);

/**
 * resize_bilinear_window() on a decoded sub-rectangle of the source.
 *
 * `region` holds the source pixels starting at (region_x, region_y) of an
 * image of full_width x full_height. Filter coefficients come from the full
 * dimensions, so the result is identical to resizing the whole image, provided
 * the region covers resize_source_window(); otherwise the result is empty.
 */
Image8 resize_bilinear_window(const Image8 &region, int region_x, int region_y,
                              int full_width, int full_height, int out_width,
                              int out_height, int window_x, int window_y,
                              int window_width, int window_height);

} // namespace dataset
} // namespace deeptile

#endif // DEEPTILE_DATASET_RESIZE_H_
