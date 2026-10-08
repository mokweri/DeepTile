/**
 * @file common_defs_bridge_check.cpp
 * @brief Checks the native architecture view against the HLS ABI definitions.
 *
 * Both views source their integer constants from arch_config.h. Compiling this
 * file together verifies that the native aliases and HLS-derived geometry stay
 * compatible without adding Xilinx dependencies to native compiler targets.
 */

#include "common_defs.h"
#include "deeptile/tile/hardware_spec.hpp"

namespace hw = deeptile::tile::hw_defaults;

static_assert(hw::kPSpatial == P_SPATIAL, "kPSpatial alias mismatch");
static_assert(hw::kPIc == P_IC, "kPIc alias mismatch");
static_assert(hw::kIcTileSize == IC_TILE_SIZE,
              "kIcTileSize alias mismatch");
static_assert(hw::kOcTileSize == OC_TILE_SIZE,
              "kOcTileSize alias mismatch");

static_assert(hw::kWeightBufDepth == WEIGHT_BUF_DEPTH,
              "kWeightBufDepth alias mismatch");
static_assert(hw::kK1CacheDepth == IFM_CACHE_DEPTH,
              "kK1CacheDepth alias mismatch");
static_assert(hw::kPsumDepth == PSUM_DEPTH, "kPsumDepth alias mismatch");

static_assert(hw::kMaxTileH == MAX_TILE_H, "kMaxTileH alias mismatch");
static_assert(hw::kMaxTileW == MAX_TILE_W, "kMaxTileW alias mismatch");
static_assert(hw::kKMax == K_MAX, "kKMax alias mismatch");
static_assert(hw::kPaddedTileH == PADDED_TILE_H,
              "kPaddedTileH alias mismatch");
static_assert(hw::kPaddedTileW == PADDED_TILE_W,
              "kPaddedTileW alias mismatch");
static_assert(hw::kMaxSpatialTile == MAX_SPATIAL_TILE_SIZE,
              "kMaxSpatialTile alias mismatch");
static_assert(hw::kGapScaleFracBits == GAP_SCALE_FRAC_BITS,
              "kGapScaleFracBits alias mismatch");

// graph_compiler.cpp aligns the per-CIG cache stride to hw::kPSpatial and bounds
// it by hw::kK1CacheDepth; the kernel does the same with P_SPATIAL_LOG2 and
// IFM_CACHE_DEPTH. If these two diverge, cache-fit decisions and cache row
// addressing disagree — a silent wrong-output class, not a crash.
static_assert((1 << P_SPATIAL_LOG2) == hw::kPSpatial,
              "P_SPATIAL_LOG2 disagrees with the scheduler's kPSpatial");
static_assert(hw::kMaxSpatialTile / hw::kPSpatial == PSUM_DEPTH,
              "scheduler psum depth derivation disagrees with PSUM_DEPTH");

int main() { return 0; }
