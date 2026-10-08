/**
 * @file hardware_spec.hpp
 * @brief Host-visible model of accelerator capabilities and limits.
 * @ingroup deeptile_compiler
 *
 * Defaults come from arch_config.h, which is shared with the kernel without
 * introducing HLS header dependencies into native tools.
 */

#pragma once

#include "arch_config.h"

#include <string>

namespace deeptile {
namespace tile {

namespace hw_defaults {

constexpr int kPSpatial         = P_SPATIAL;
constexpr int kPIc              = P_IC;
constexpr int kIcTileSize       = IC_TILE_SIZE;
constexpr int kOcTileSize       = OC_TILE_SIZE;
constexpr int kWeightBufDepth   = WEIGHT_BUF_DEPTH;   ///< Per weight bank.
constexpr int kBiasBufDepth     = BIAS_BUF_DEPTH;
constexpr int kK1CacheDepth     = IFM_CACHE_DEPTH;
constexpr int kMaxTileH         = MAX_TILE_H;
constexpr int kMaxTileW         = MAX_TILE_W;
constexpr int kKMax             = K_MAX;
constexpr int kPaddedTileH      = PADDED_TILE_H;
constexpr int kPaddedTileW      = PADDED_TILE_W;
constexpr int kMaxSpatialTile   = MAX_SPATIAL_TILE_SIZE;
constexpr int kGapScaleFracBits = GAP_SCALE_FRAC_BITS;
constexpr int kPsumDepth        = PSUM_DEPTH;
constexpr int kDwMaxCogs3       = DW_MAX_COGS;     ///< Fused 3x3 depthwise COG blocks.
constexpr int kDwMaxCogs5       = DW_MAX_COGS_5;   ///< Fused 5x5 depthwise COG blocks.
constexpr int kDwWordsPerCog3   = DW_WORDS_PER_COG_3;
constexpr int kDwWordsPerCog5   = DW_WORDS_PER_COG_5;

} // namespace hw_defaults

/**
 * @brief Capability/limit model for one accelerator configuration.
 */
struct HardwareSpec {
  int bitwidth       = BITWIDTH;
  int gap_scale_frac_bits = hw_defaults::kGapScaleFracBits;
  int p_spatial      = hw_defaults::kPSpatial;
  int p_ic           = hw_defaults::kPIc;
  int ic_tile_size   = hw_defaults::kIcTileSize;
  int oc_tile_size   = hw_defaults::kOcTileSize;
  int weight_buf_depth = hw_defaults::kWeightBufDepth;  ///< Per ping-pong bank.
  int bias_buf_depth   = hw_defaults::kBiasBufDepth;
  int psum_depth     = hw_defaults::kPsumDepth;
  int k1_cache_depth = hw_defaults::kK1CacheDepth;
  int max_tile_h     = hw_defaults::kMaxTileH;
  int max_tile_w     = hw_defaults::kMaxTileW;
  int k_max          = hw_defaults::kKMax;
  int max_spatial_tile = hw_defaults::kMaxSpatialTile;

  bool supports_residual_add = true;
  bool supports_maxpool      = true;
  bool supports_gap          = true;
  bool supports_depthwise    = true;

  /// Structural MAC throughput per cycle.
  int macs_per_cycle() const { return p_spatial * p_ic * oc_tile_size; }

  /// Structural peak in GOPS (2 ops/MAC) at a given clock.
  double peak_gops(double clock_mhz) const {
    return 2.0 * static_cast<double>(macs_per_cycle()) * clock_mhz / 1000.0;
  }

  /// PSUM depth implied by a spatial-parallelism factor (one psum row per group).
  static int psum_depth_for(int spatial_tile, int p_spatial_) {
    return p_spatial_ > 0 ? spatial_tile / p_spatial_ : 0;
  }

  /// Return a variant with dependent partial-sum depth recomputed.
  HardwareSpec with_p_spatial(int new_p_spatial) const {
    HardwareSpec s = *this;
    s.p_spatial  = new_p_spatial;
    s.psum_depth = psum_depth_for(max_spatial_tile, new_p_spatial);
    return s;
  }
};

/** Canonical compatibility key for manifest fingerprinting. */
inline std::string architecture_key(const HardwareSpec &hw) {
  return "bitwidth=" + std::to_string(hw.bitwidth) +
         ";gap_scale_frac_bits=" +
         std::to_string(hw.gap_scale_frac_bits) +
         ";ic_tile_size=" + std::to_string(hw.ic_tile_size) +
         ";oc_tile_size=" + std::to_string(hw.oc_tile_size) +
         ";p_ic=" + std::to_string(hw.p_ic) +
         ";p_spatial=" + std::to_string(hw.p_spatial) +
         ";max_tile_height=" + std::to_string(hw.max_tile_h) +
         ";max_tile_width=" + std::to_string(hw.max_tile_w) +
         ";max_kernel_size=" + std::to_string(hw.k_max) +
         ";ifm_cache_depth=" + std::to_string(hw.k1_cache_depth) +
         ";weight_buffer_depth=" + std::to_string(hw.weight_buf_depth) +
         ";bias_buffer_depth=" + std::to_string(hw.bias_buf_depth);
}

/** Existing cost-model fingerprint format, centralized with HardwareSpec. */
inline std::string cost_model_architecture_key(const HardwareSpec &hw) {
  return "P_SPATIAL" + std::to_string(hw.p_spatial) +
         "_OC_TILE_SIZE" + std::to_string(hw.oc_tile_size) +
         "_IC_TILE_SIZE" + std::to_string(hw.ic_tile_size) +
         "_P_IC" + std::to_string(hw.p_ic) +
         "_MAX_TILE_H" + std::to_string(hw.max_tile_h) +
         "_MAX_TILE_W" + std::to_string(hw.max_tile_w);
}

static_assert(hw_defaults::kMaxSpatialTile ==
                  hw_defaults::kMaxTileH * hw_defaults::kMaxTileW,
              "max_spatial_tile must equal max_tile_h * max_tile_w");
static_assert(hw_defaults::kPsumDepth * hw_defaults::kPSpatial ==
                  hw_defaults::kMaxSpatialTile,
              "psum_depth must equal max_spatial_tile / p_spatial");
// INVARIANT: Kernel pixel/group addressing uses shift-and-mask operations.
static_assert(hw_defaults::kPSpatial > 0 &&
                  (hw_defaults::kPSpatial & (hw_defaults::kPSpatial - 1)) == 0,
              "p_spatial must be a power of two");
// INVARIANT: Raw-IFM column-parity banking requires at least two spatial lanes.
static_assert(hw_defaults::kPSpatial >= 2,
              "p_spatial=1 requires a dedicated single-bank IFM loader");

} // namespace tile
} // namespace deeptile
