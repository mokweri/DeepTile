/**
 * @file stem_fold.h
 * @brief Space-to-depth rewrite of a stride-2 RGB stem convolution.
 * @ingroup deeptile_host
 *
 * A network's first convolution sees three input channels, so it fills three
 * of the array's sixteen input lanes, and at stride 2 its input loader moves
 * four input pixels per output pixel. Folding each 2x2 block of the
 * zero-padded input into channels turns a KxK stride-2 convolution over
 * [C, H, W] into an equivalent ceil(K/2) x ceil(K/2) stride-1 convolution over
 * [4C, Hp/2, Wp/2] with no padding: the same products, summed in a different
 * order, so the integer result is identical. The kernel is unchanged; the
 * rewrite is made to the graph before compilation, and the host lays the input
 * out in the folded form. On P4 it makes the ResNet stem 3.2x and the
 * MobileNet-class stem 2.3x faster and cuts the input handoff from 802 KB to
 * about 212 KB.
 */

#ifndef DEEPTILE_RUNTIME_STEM_FOLD_H_
#define DEEPTILE_RUNTIME_STEM_FOLD_H_

#include "deeptile/compiler/graph_types.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace deeptile {
namespace runtime {

/** Geometry of an applied stem fold. Inactive when no stem qualified. */
struct StemFold {
  bool active = false;
  std::string input_id;  ///< Graph input that is folded.
  std::string node_id;   ///< Stem convolution that consumes it.
  int channels = 0;      ///< Original input [C, H, W].
  int height = 0;
  int width = 0;
  int pad = 0;           ///< Zero border added before folding.
  int folded_height = 0; ///< Folded input is [4C, folded_height, folded_width].
  int folded_width = 0;
};

/** Raw int8 data that replaces a tensor's file contents. */
using TensorOverrides = std::map<std::string, std::vector<std::int8_t>>;

/** True unless DEEPTILE_STEM_S2D=0 disables the rewrite (on by default). */
bool stem_fold_requested();

/**
 * Rewrite the graph's stem in place if it qualifies.
 *
 * A stem qualifies when the graph has one input of at most four channels,
 * consumed only by a dense convolution with an odd kernel of at most 7,
 * stride 2, symmetric padding and no dilation, whose weights nothing else
 * uses. The input and weight tensors take their folded shapes, and their
 * folded contents are returned in `overrides` (the input only if the package
 * bundles one). Nothing changes when the stem does not qualify.
 *
 * @return False only if a qualifying stem's tensors could not be read.
 */
bool fold_stem(deeptile::graph::GraphSpec &graph, StemFold &fold,
               TensorOverrides &overrides, std::string &error);

/** Fold an original [C, H, W] input into [4C, folded_height, folded_width]. */
std::vector<std::int8_t> fold_input_chw(const std::vector<std::int8_t> &chw,
                                        const StemFold &fold);

/**
 * Lane of the folded input that holds channel `c` of the source pixel at
 * offset (dy, dx) within its 2x2 block. Shared by the weight rewrite, the
 * input fold and the dataset preprocessing, which must agree.
 */
inline int stem_fold_lane(int dy, int dx, int c, int channels) {
  return (dy * 2 + dx) * channels + c;
}

} // namespace runtime
} // namespace deeptile

#endif // DEEPTILE_RUNTIME_STEM_FOLD_H_
